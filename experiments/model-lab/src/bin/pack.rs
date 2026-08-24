//! pack: convert HF GPT-2 safetensors dir -> SMT v2-lite packs (f16 canonical, q8 atom).
#![allow(dead_code)]
use model_lab::atoms::{f32_to_f16, q8_encode, ATOM_F16, ATOM_Q8};
use model_lab::format::{SectionWriter, TensorRecord, SEC_GRAPH, SEC_META, SEC_TENSORS, SEC_TOKENIZER};
use serde_json::Value;
use std::collections::BTreeMap;

fn main() {
    let dir = "assets";
    let cfg: Value = serde_json::from_slice(
        &std::fs::read(format!("{dir}/config.json")).expect("config.json"),
    )
    .unwrap();
    let tok_bytes = std::fs::read(format!("{dir}/tokenizer.json")).expect("tokenizer.json");

    // ---- parse safetensors ----
    let raw = std::fs::read(format!("{dir}/model.safetensors")).expect("safetensors");
    let n = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    let header: BTreeMap<String, Value> = serde_json::from_slice(&raw[8..8 + n]).unwrap();
    let data = 8 + n;
    println!("safetensors header entries: {}", header.len() - 1);

    // ---- build tensor payloads under two atoms ----
    struct Out {
        recs: Vec<TensorRecord>,
        payload: Vec<u8>,
    }
    let mut build = |atom: &str| -> Out {
        let mut out = Out { recs: vec![], payload: vec![] };
        for (name, spec) in header.iter() {
            if name == "__metadata__" {
                continue;
            }
            assert_eq!(spec["dtype"].as_str(), Some("F32"), "{name}");
            let shape: Vec<u32> = spec["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as u32)
                .collect::<Vec<_>>();
            let [o0, o1] = [spec["data_offsets"][0].as_u64().unwrap() as usize,
                            spec["data_offsets"][1].as_u64().unwrap() as usize];
            let mut f32s: Vec<f32> = raw[data + o0..data + o1]
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            // HF GPT-2 Conv1D stores [in,out]; transpose to [out,in] at CONVERT time
            // (SMT doctrine: execution layout is decided by the converter, not the runtime).
            let suffixes = [
                "attn.c_attn.weight",
                "attn.c_proj.weight",
                "mlp.c_fc.weight",
                "mlp.c_proj.weight",
            ];
            let mut shape = shape.clone();
            if shape.len() == 2 && suffixes.iter().any(|s| name.ends_with(s)) {
                let (a, b) = (shape[0] as usize, shape[1] as usize);
                let mut t = vec![0f32; a * b];
                for i in 0..a {
                    for j in 0..b {
                        t[j * a + i] = f32s[i * b + j];
                    }
                }
                std::mem::swap(&mut f32s, &mut t);
                shape = vec![b as u32, a as u32];
            }
            let numel: usize = shape.iter().map(|&x| x as usize).product();
            // big 2D tensors -> Q8; small/1D (biases, layernorm) -> F16
            let (atom_name, bytes) =
                if atom == "auto" && numel >= 4096 && shape.len() == 2 {
                    (ATOM_Q8.to_string(), q8_encode(&f32s))
                } else if atom == "auto" || atom == ATOM_F16 {
                    (
                        ATOM_F16.to_string(),
                        f32s.iter().map(|v| f32_to_f16(*v).to_le_bytes()).flatten().collect(),
                    )
                } else if atom == ATOM_Q8 && shape.len() == 2 {
                    (ATOM_Q8.to_string(), q8_encode(&f32s))
                } else {
                    (
                        ATOM_F16.to_string(),
                        f32s.iter().map(|v| f32_to_f16(*v).to_le_bytes()).flatten().collect(),
                    )
                };
            let digest: [u8; 16] = blake3::hash(&bytes).as_bytes()[..16].try_into().unwrap();
            out.recs.push(TensorRecord {
                name: name.clone(),
                shape,
                atom: atom_name,
                offset: out.payload.len() as u64,
                len: bytes.len() as u64,
                digest,
            });
            out.payload.extend_from_slice(&bytes);
        }
        out
    };
    let q8 = build("auto");
    let f16 = build(ATOM_F16);
    drop(raw);

    // ---- graph-as-data (op list the executor resolves) ----
    let n_layer = cfg["n_layer"].as_u64().unwrap();
    let mut graph: Vec<String> = vec!["embed".into(), "pos_add".into()];
    for l in 0..n_layer {
        graph.push(format!("h{l}.ln_1"));
        graph.push(format!("h{l}.attn.c_attn"));
        graph.push(format!("h{l}.attn.mha_causal"));
        graph.push(format!("h{l}.attn.c_proj"));
        graph.push(format!("h{l}.resid"));
        graph.push(format!("h{l}.ln_2"));
        graph.push(format!("h{l}.mlp.c_fc"));
        graph.push(format!("h{l}.gelu"));
        graph.push(format!("h{l}.mlp.c_proj"));
        graph.push(format!("h{l}.resid"));
    }
    graph.push("ln_f".into());
    graph.push("tied_head".into());

    for (tag, out) in [("q8", &q8), ("f16", &f16)] {
        let path = format!("assets/gpt2-{tag}.smt");
        let t0 = std::time::Instant::now();
        let mut w = SectionWriter::new(std::io::BufWriter::with_capacity(
            1 << 24,
            std::fs::File::create(&path).unwrap(),
        ))
        .unwrap();

        let meta = serde_json::json!({
            "arch": {
                "name": "gpt2",
                "n_embd": cfg["n_embd"], "n_layer": cfg["n_layer"], "n_head": cfg["n_head"],
                "n_ctx": cfg["n_ctx"], "vocab_size": cfg["vocab_size"],
                "ln_eps": cfg["layer_norm_epsilon"],
            },
            "atom_manifest": out.recs.iter().map(|r| (&r.name, &r.atom)).collect::<BTreeMap<_, _>>(),
            "provenance": {"source": "openai-community/gpt2 model.safetensors", "converter": "model-lab/pack 0.1"},
        });
        w.section(SEC_META, 0, &serde_json::to_vec(&meta).unwrap()).unwrap();
        w.section(SEC_GRAPH, 0, &serde_json::to_vec(&graph).unwrap()).unwrap();
        w.section(SEC_TOKENIZER, 0, &tok_bytes).unwrap();

        // TENSORS section: [u32 count][u32 json_len][records json][payloads]
        let recs_json = serde_json::to_vec(&out.recs).unwrap();
        let mut sec = Vec::with_capacity(8 + recs_json.len() + out.payload.len());
        sec.extend_from_slice(&(out.recs.len() as u32).to_le_bytes());
        sec.extend_from_slice(&(recs_json.len() as u32).to_le_bytes());
        sec.extend_from_slice(&recs_json);
        sec.extend_from_slice(&out.payload);
        w.section(SEC_TENSORS, 0, &sec).unwrap();

        let (cid, size) = w.finish().unwrap();
        println!(
            "{path}: {} tensors, {:.1} MB, {:?}, content_id blake3:{}",
            out.recs.len(),
            size as f64 / 1048576.0,
            t0.elapsed(),
            hex(&cid[..8])
        );
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{x:02x}")).collect()
    }
}
