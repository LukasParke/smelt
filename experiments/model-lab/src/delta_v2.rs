//! Delta v2 file format for routed multi-site adapters (Agent C).
//!
//! SectionWriter-based SMT file:
//!   META     — json {kind:"adapter_v2", format_version, base_content_id, generation, sites:[…]}
//!              with per-site descriptors incl. routes (fact_id/start/end) and a generation counter.
//!   TENSORS  — [u32 count][u32 json_len][records JSON][payload bytes]; ONE record per site named
//!              "site.{i}" whose payload is concat(A_slice-major bytes, B column-major bytes),
//!              atom "core.f32.raw", digest = blake3(payload)[..16].
//!
//! Loader validates (a) the merkle container, (b) base content_id binding against the engine's
//! pack, (c) every per-site record digest. Any mismatch is a hard error.
#![allow(dead_code)]

use crate::adapter_v2::{
    cid_hex, digest16, f32_from_le, f32_le_bytes, meta_json, AdapterV2, RouteSpec, SiteAdapter,
    SiteKind,
};
use crate::format::{PackReader, SectionWriter, TensorRecord, SEC_META, SEC_TENSORS};
use crate::gpt2::Engine;

/// Persist the adapter as a delta v2 bound to `base_cid`.
pub fn save(path: &str, base_cid: &[u8; 32], ad: &AdapterV2) -> std::io::Result<()> {
    save_gen(path, base_cid, ad, 0)
}

/// Same, with explicit generation counter (used by consolidation to bump generations).
pub fn save_gen(path: &str, base_cid: &[u8; 32], ad: &AdapterV2, generation: u32) -> std::io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut w = SectionWriter::new(std::io::BufWriter::with_capacity(1 << 20, f))?;
    w.section(SEC_META, 0, &serde_json::to_vec(&meta_json(&cid_hex(base_cid), generation, ad)).unwrap())?;

    // one record per site: payload = concat(A bytes, B bytes)
    let mut recs: Vec<TensorRecord> = Vec::with_capacity(ad.sites.len());
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(ad.sites.len());
    let mut off: u64 = 0;
    for (i, s) in ad.sites.iter().enumerate() {
        let mut body = f32_le_bytes(&s.a);
        body.extend_from_slice(&f32_le_bytes(&s.b));
        let rec = TensorRecord {
            name: format!("site.{i}"),
            shape: vec![(s.a.len() + s.b.len()) as u32],
            atom: "core.f32.raw".into(),
            offset: off,
            len: body.len() as u64,
            digest: digest16(&body),
        };
        off += body.len() as u64;
        payloads.push(body);
        recs.push(rec);
    }
    let recs_json = serde_json::to_vec(&recs).unwrap();
    let mut sec = Vec::new();
    sec.extend_from_slice(&(recs.len() as u32).to_le_bytes());
    sec.extend_from_slice(&(recs_json.len() as u32).to_le_bytes());
    sec.extend_from_slice(&recs_json);
    for p in &payloads {
        sec.extend_from_slice(p);
    }
    w.section(SEC_TENSORS, 0, &sec)?;
    w.finish()?;
    Ok(())
}

/// Load + validate a delta v2 against the engine's base pack.
/// Errors on: bad container, missing sections, base-cid binding mismatch, digest mismatch,
/// malformed site descriptor or route.
pub fn load(path: &str, eng: &Engine) -> Result<AdapterV2, String> {
    let pack = PackReader::open(path).map_err(|e| format!("delta open: {e:?}"))?;
    pack.verify().map_err(|e| format!("delta container verify failed: {e:?}"))?;

    let mj: serde_json::Value = serde_json::from_slice(
        pack.section(SEC_META).ok_or("delta missing META section")?,
    )
    .map_err(|e| format!("delta META parse: {e}"))?;

    if mj["kind"].as_str() != Some("adapter_v2") {
        return Err(format!("not an adapter_v2 delta (kind={:?})", mj["kind"].as_str()));
    }

    // --- base cid binding ---
    let eng_hex = cid_hex(&eng.pack.content_id());
    match mj["base_content_id"].as_str() {
        Some(h) if h == eng_hex => {}
        Some(h) => return Err(format!("delta binds to base {h} but engine is {eng_hex}")),
        None => return Err("delta META missing base_content_id".into()),
    }

    // --- site descriptors ---
    let sites_v = mj["sites"].as_array().ok_or("delta META missing sites array")?;
    if sites_v.is_empty() {
        return Err("delta has zero sites".into());
    }

    // --- tensor records ---
    let t = pack.section(SEC_TENSORS).ok_or("delta missing TENSORS section")?;
    if t.len() < 8 {
        return Err("delta TENSORS truncated header".into());
    }
    let count = u32::from_le_bytes(t[..4].try_into().unwrap()) as usize;
    let json_len = u32::from_le_bytes(t[4..8].try_into().unwrap()) as usize;
    let recs: Vec<TensorRecord> =
        serde_json::from_slice(&t[8..8 + json_len]).map_err(|e| format!("records parse: {e}"))?;
    if count != recs.len() || count != sites_v.len() {
        return Err(format!(
            "record/site/count mismatch: {count} records, {} site descriptors",
            sites_v.len()
        ));
    }
    let data0 = 8 + json_len;

    let mut sites = Vec::with_capacity(count);
    for (i, sv) in sites_v.iter().enumerate() {
        let kind = SiteKind::from_str(sv["kind"].as_str().ok_or("site missing kind")?)?;
        let layer = sv["layer"].as_u64().ok_or("site missing layer")? as usize;
        let d = sv["d"].as_u64().ok_or("site missing d")? as usize;
        let r = sv["r"].as_u64().ok_or("site missing r")? as usize;
        let mut cols = Vec::new();
        for rc in sv["route"].as_array().ok_or("site missing route")? {
            cols.push((
                rc["fact_id"].as_u64().ok_or("route missing fact_id")?,
                rc["start"].as_u64().ok_or("route missing start")? as usize,
                rc["end"].as_u64().ok_or("route missing end")? as usize,
            ));
        }
        let mut site = SiteAdapter::new(kind, layer, d, RouteSpec { cols });
        if site.r != r {
            return Err(format!("site[{i}] META r={r} but route sums to {}", site.r));
        }

        // fetch record i, validate digest, dequantize
        let rec = &recs[i];
        let s = data0 + rec.offset as usize;
        let e = s + rec.len as usize;
        if e > t.len() {
            return Err(format!("record {} out of range", rec.name));
        }
        let body = &t[s..e];
        let dg = digest16(body);
        if dg != rec.digest {
            return Err(format!("digest mismatch on {}", rec.name));
        }
        if body.len() != (site.a.len() + site.b.len()) * 4 {
            return Err(format!(
                "record {} payload {} B != expected {} B",
                rec.name,
                body.len(),
                (site.a.len() + site.b.len()) * 4
            ));
        }
        let vals = f32_from_le(body);
        let (na, nb) = (site.a.len(), site.b.len());
        site.a.copy_from_slice(&vals[..na]);
        site.b.copy_from_slice(&vals[na..na + nb]);
        sites.push(site);
    }
    Ok(AdapterV2 { sites })
}

/// Read the raw META json of a delta file (for tooling/tests: inspect generation counter etc.).
pub fn meta_of(path: &str) -> Result<serde_json::Value, String> {
    let pack = PackReader::open(path).map_err(|e| format!("delta open: {e:?}"))?;
    pack.verify().map_err(|e| format!("delta container verify failed: {e:?}"))?;
    let sec = pack.section(SEC_META).ok_or("delta missing META section")?;
    serde_json::from_slice(sec).map_err(|e| format!("delta META parse: {e}"))
}
