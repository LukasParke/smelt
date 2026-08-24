//! GPT-2-style BPE loaded from a HF tokenizer.json (vocab + merges + byte-level map).
//! Hand-rolled pretokenizer matching the classic GPT-2 split pattern.
#![allow(dead_code)]
use serde_json::Value;
use std::collections::HashMap;

pub struct Bpe {
    pub ranks_count: usize,
    pub vocab: HashMap<String, u32>,
    ranks: HashMap<(String, String), u32>,
    byte_enc: Vec<String>,          // 256 byte -> unicode-char string
    byte_dec: HashMap<char, u8>,
}

fn bytes_to_unicode() -> (Vec<String>, HashMap<char, u8>) {
    // The canonical GPT-2 byte<->unicode table.
    let mut bs: Vec<u8> = (b'!'..=b'~').chain(b'\xA1'..=b'\xAC').chain(b'\xAE'..=b'\xFF').collect();
    let mut cs: Vec<char> = bs.iter().map(|&b| b as char).collect();
    let mut n = 0u32;
    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(char::from_u32(256 + n).unwrap());
            n += 1;
        }
    }
    let mut enc = vec![String::new(); 256];
    let mut dec = HashMap::new();
    for (b, c) in bs.into_iter().zip(cs.into_iter()) {
        enc[b as usize] = c.to_string();
        dec.insert(c, b);
    }
    (enc, dec)
}

impl Bpe {
    pub fn load(tokenizer_json: &[u8]) -> Self {
        let v: Value = serde_json::from_slice(tokenizer_json).expect("tokenizer.json parse");
        let model = &v["model"];
        assert!(model.get("vocab").map(|v| v.is_object()).unwrap_or(false), "expected BPE vocab");
        let mut vocab = HashMap::new();
        for (tok, id) in model["vocab"].as_object().expect("vocab") {
            vocab.insert(tok.clone(), id.as_u64().unwrap() as u32);
        }
        let mut ranks = HashMap::new();
        for (i, m) in model["merges"].as_array().expect("merges").iter().enumerate() {
            let pair = match m {
                Value::Array(a) => (
                    a[0].as_str().unwrap().to_string(),
                    a[1].as_str().unwrap().to_string(),
                ),
                Value::String(s) => {
                    let mut it = s.split(' ');
                    (
                        it.next().unwrap().to_string(),
                        it.next().unwrap_or("").to_string(),
                    )
                }
                _ => continue,
            };
            ranks.insert(pair, i as u32);
        }
        let rc = ranks.len();
        let (byte_enc, byte_dec) = bytes_to_unicode();
        Self { ranks_count: rc, vocab, ranks, byte_enc, byte_dec }
    }

    fn chunk_to_symbols(&self, chunk: &str) -> Vec<String> {
        chunk
            .as_bytes()
            .iter()
            .map(|&b| self.byte_enc[b as usize].clone())
            .collect()
    }

    /// Classic lowest-rank BPE merge over one word's symbols.
    fn bpe_word(&self, word: &str) -> Vec<u32> {
        let mut parts = self.chunk_to_symbols(word);
        if parts.len() == 1 {
            return vec![self.vocab[&parts[0]]];
        }
        loop {
            let mut best: Option<(u32, usize)> = None;
            for i in 0..parts.len() - 1 {
                if let Some(r) = self.ranks.get(&(parts[i].clone(), parts[i + 1].clone())) {
                    if best.map(|(br, _)| *r < br).unwrap_or(true) {
                        best = Some((*r, i));
                    }
                }
            }
            let (_, i) = match best {
                Some(b) => b,
                None => break,
            };
            let merged = format!("{}{}", parts[i], parts[i + 1]);
            parts[i] = merged;
            parts.remove(i + 1);
            if parts.len() == 1 {
                break;
            }
        }
        parts.iter().map(|p| self.vocab[p]).collect()
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();
        for chunk in self.pretokenize(text) {
            ids.extend(self.bpe_word(&chunk));
        }
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        // id -> token string -> unicode chars -> bytes
        let inv: HashMap<u32, &String> = self.vocab.iter().map(|(k, v)| (*v, k)).collect();
        let mut bytes = Vec::new();
        for &id in ids {
            if let Some(tok) = inv.get(&id) {
                for c in tok.chars() {
                    bytes.push(self.byte_dec.get(&c).copied().unwrap_or(b'?'));
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// GPT-2 pretokenizer without a regex dependency:
    /// 's|'t|'re|'ve|'m|'ll|'d| ?letters+| ?numbers+| ?other-nonspace+|\s+(?!\S)|\s+
    fn pretokenize<'a>(&self, text: &'a str) -> Vec<&'a str> {
        let cs: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < cs.len() {
            // contractions (case-sensitive, as the reference pattern)
            let rest: String = cs[i..].iter().take(4).collect();
            let con = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d"]
                .iter()
                .find(|c| rest.starts_with(**c));
            if let Some(c) = con {
                out.push(&text[char_off(&cs, i)..char_off(&cs, i + c.chars().count())]);
                i += c.chars().count();
                continue;
            }
            let start = i;
            let mut lead_space = cs[i] == ' ';
            if lead_space {
                i += 1;
                if i == cs.len() {
                    break; // trailing single space handled below as whitespace run
                }
            }
            let c = cs[i];
            if c.is_alphabetic() {
                while i < cs.len() && cs[i].is_alphabetic() {
                    i += 1;
                }
                push(&cs, start, i, &mut out, text);
                continue;
            }
            if c.is_numeric() {
                while i < cs.len() && cs[i].is_numeric() {
                    i += 1;
                }
                push(&cs, start, i, &mut out, text);
                continue;
            }
            if !lead_space && c != ' ' && !c.is_whitespace() || (!lead_space && c == ' ') {
                // other non-space run (includes punctuation)
                if c == ' ' {
                    // whitespace run started without letter/number following
                    i = start;
                }
                let mut j = i;
                while j < cs.len() && cs[j] != ' ' && !cs[j].is_whitespace() {
                    j += 1;
                }
                if j > i {
                    push(&cs, start, j, &mut out, text);
                    i = j;
                    continue;
                }
            }
            // whitespace run: consume spaces; last space belongs to next word if followed by non-space
            i = start;
            let mut j = i;
            while j < cs.len() && cs[j] == ' ' {
                j += 1;
            }
            if j < cs.len() && !cs[j].is_whitespace() && j > i {
                // "\s+(?!\S)" fails => emit all but last space; " ?X+" grabs the last
                if j - i > 1 {
                    push(&cs, i, j - 1, &mut out, text);
                    i = j - 1;
                } else {
                    i = j - 1;
                    // fallthrough: next loop treats " X"
                    if i == start {
                        // single leading space consumed by next-word branch next iteration
                        i = start + 1; // skip the space; next branch re-reads it as lead
                        continue;
                    }
                }
                continue;
            }
            // pure whitespace (incl newlines) run
            while j < cs.len() && cs[j].is_whitespace() {
                j += 1;
            }
            push(&cs, i, j, &mut out, text);
            i = j;
        }
        out.retain(|s| !s.is_empty());
        out
    }
}

fn push<'a>(cs: &[char], a: usize, b: usize, out: &mut Vec<&'a str>, text: &'a str) {
    if b > a {
        out.push(&text[char_off(cs, a)..char_off(cs, b)]);
    }
}
fn char_off(cs: &[char], idx: usize) -> usize {
    cs[..idx].iter().map(|c| c.len_utf8()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    #[test]
    fn debug_chain() {
        let raw = fs::read(concat!(env!("CARGO_MANIFEST_DIR"), "/assets/tokenizer.json")).unwrap();
        let b = Bpe::load(&raw);
        let pt = b.pretokenize("Once upon a time");
        println!("chunks: {:?}", pt);
        let parts: Vec<String> = b.chunk_to_symbols("Once");
        println!("symbols: {:?}", parts);
        println!("rank On: {:?}", b.ranks.get(&("O".to_string(), "n".to_string())));
        println!("be[79]={:?} be[110]={:?} be[99]={:?} idx_of_O={:?})", b.byte_enc[79], b.byte_enc[110], b.byte_enc[99], b.byte_enc.iter().position(|s| s=="O"));
        println!("bpe_word(Once): {:?} -> ids", b.bpe_word("Once"));
        assert_eq!(b.encode("Once"), vec![7454u32]);
    }
}
