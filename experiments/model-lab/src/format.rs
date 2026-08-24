//! SMT container format (v2-lite): superheader + TLV sections + content addressing.
//! Implements the spec's load-bearing properties: tagged sections, per-tensor BLAKE3 digests,
//! merkle content_id, atom-tagged tensors, graph-as-data, embedded tokenizer.
#![allow(dead_code)]

pub const MAGIC: &[u8; 4] = b"SMT\x01";
pub const VERSION: u16 = 2;

pub const SEC_INDEX: u32 = 1;
pub const SEC_META: u32 = 2;
pub const SEC_GRAPH: u32 = 3;
pub const SEC_TOKENIZER: u32 = 4;
pub const SEC_TENSORS: u32 = 5;

use std::io::{self, Write};

/// One tensor record in the TENSORS section table.
#[derive(serde::Serialize, serde::Deserialize, Clone)]
pub struct TensorRecord {
    pub name: String,
    pub shape: Vec<u32>,
    /// namespaced atom ref: "core.f16" | "core.i8.b32.f16scale"
    pub atom: String,
    pub offset: u64,
    pub len: u64,
    #[serde(with = "hex16")]
    pub digest: [u8; 16],
}

pub mod hex16 {
    use serde::{Deserialize, Deserializer, Serializer};
    pub fn serialize<S: Serializer>(d: &[u8; 16], s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&d.iter().map(|b| format!("{b:02x}")).collect::<String>())
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(s: D) -> Result<[u8; 16], D::Error> {
        let t = String::deserialize(s)?;
        let mut o = [0u8; 16];
        for i in 0..16 {
            o[i] = u8::from_str_radix(&t[2 * i..2 * i + 2], 16).map_err(serde::de::Error::custom)?;
        }
        Ok(o)
    }
}

/// Section framing: type u32 | flags u32 | len u64 | payload
pub struct SectionWriter<W: io::Write> {
    inner: W,
    digests: Vec<(u32, [u8; 32])>,
}
impl<W: io::Write + io::Seek> SectionWriter<W> {
    pub fn new(mut inner: W) -> io::Result<Self> {
        // placeholder superheader (128 B), patched by finish()
        let mut hdr = vec![0u8; 128];
        hdr[..4].copy_from_slice(MAGIC);
        hdr[4..6].copy_from_slice(&VERSION.to_le_bytes());
        inner.write_all(&hdr)?;
        Ok(Self { inner, digests: vec![] })
    }
    pub fn section(&mut self, ty: u32, flags: u32, payload: &[u8]) -> io::Result<()> {
        self.inner.write_all(&ty.to_le_bytes())?;
        self.inner.write_all(&flags.to_le_bytes())?;
        self.inner.write_all(&(payload.len() as u64).to_le_bytes())?;
        self.inner.write_all(payload)?;
        self.digests.push((ty, blake3::hash(payload).into()));
        Ok(())
    }
    /// INDEX section listing all prior sections, then patch superheader with content_id.
    pub fn finish(mut self) -> io::Result<([u8; 32], u64)> {
        let mut idx = Vec::new();
        for (ty, d) in &self.digests {
            idx.extend_from_slice(&ty.to_le_bytes());
            idx.extend_from_slice(d);
        }
        self.section(SEC_INDEX, 0, &idx)?;
        // merkle over sorted section digests
        let mut sorted: Vec<[u8; 32]> = self.digests.iter().map(|(_, d)| *d).collect();
        sorted.sort();
        let mut hasher = blake3::Hasher::new();
        for d in sorted {
            hasher.update(&d);
        }
        let content_id: [u8; 32] = hasher.finalize().into();
        let end = self.inner.seek(io::SeekFrom::Current(0))?;
        // patch header: content_id @ 16, file_len @ 48
        self.inner.seek(io::SeekFrom::Start(16))?;
        self.inner.write_all(&content_id)?;
        self.inner.seek(io::SeekFrom::Start(48))?;
        self.inner.write_all(&end.to_le_bytes())?;
        self.inner.flush()?;
        Ok((content_id, end))
    }
}

pub struct PackReader {
    pub map: memmap2::Mmap,
}
#[derive(Debug)]
pub enum PackError {
    BadMagic,
    DigestMismatch(String),
    MissingSection(u32),
    Io(io::Error),
}
impl From<io::Error> for PackError {
    fn from(e: io::Error) -> Self {
        PackError::Io(e)
    }
}

impl PackReader {
    pub fn open(path: &str) -> Result<Self, PackError> {
        let f = std::fs::File::open(path)?;
        let map = unsafe { memmap2::Mmap::map(&f)? };
        if &map[..4] != MAGIC {
            return Err(PackError::BadMagic);
        }
        Ok(Self { map })
    }
    pub fn content_id(&self) -> [u8; 32] {
        let mut d = [0u8; 32];
        d.copy_from_slice(&self.map[16..48]);
        d
    }
    /// Verify merkle content_id against all sections present.
    pub fn verify(&self) -> Result<(), PackError> {
        let mut off = 128usize;
        let mut digests = Vec::new();
        while off < self.map.len() {
            let ty = u32::from_le_bytes(self.map[off..off + 4].try_into().unwrap());
            let _flags = u32::from_le_bytes(self.map[off + 4..off + 8].try_into().unwrap());
            let len = u64::from_le_bytes(self.map[off + 8..off + 16].try_into().unwrap()) as usize;
            let p = off + 16;
            digests.push((ty, blake3::hash(&self.map[p..p + len]).into()));
            off = p + len;
        }
        let mut sorted: Vec<[u8; 32]> = digests.into_iter().map(|(_, d)| d).collect();
        sorted.sort();
        let mut h = blake3::Hasher::new();
        for d in sorted {
            h.update(&d);
        }
        let cid: [u8; 32] = h.finalize().into();
        if cid == self.content_id() {
            Ok(())
        } else {
            Err(PackError::DigestMismatch("content_id".into()))
        }
    }
    pub fn section_offset(&self, ty: u32) -> Option<usize> {
        let mut off = 128usize;
        while off < self.map.len() {
            let t = u32::from_le_bytes(self.map[off..off + 4].try_into().unwrap());
            let len = u64::from_le_bytes(self.map[off + 8..off + 16].try_into().unwrap()) as usize;
            let p = off + 16;
            if t == ty {
                return Some(p);
            }
            off = p + len;
        }
        None
    }
    pub fn section(&self, ty: u32) -> Option<&[u8]> {
        let mut off = 128usize;
        while off < self.map.len() {
            let t = u32::from_le_bytes(self.map[off..off + 4].try_into().unwrap());
            let len = u64::from_le_bytes(self.map[off + 8..off + 16].try_into().unwrap()) as usize;
            let p = off + 16;
            if t == ty {
                return Some(&self.map[p..p + len]);
            }
            off = p + len;
        }
        None
    }
}
