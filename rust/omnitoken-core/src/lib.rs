#![allow(non_local_definitions)]
use pyo3::prelude::*;
use std::collections::HashMap;
use rkyv::{Archive, Deserialize, Serialize};
use omnitoken_code::CodePreTokenizer;

type EncodeResult = (Vec<u32>, Vec<(usize, usize)>);

#[pyclass]
#[derive(Clone, Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub struct Tokenizer {
    pub vocab: HashMap<String, f64>,
    pub vocab_to_ids: HashMap<String, u32>,
    pub ids_to_vocab: HashMap<u32, String>,
    code_pre_tokenizer: CodePreTokenizer,
}

#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new(vocab: HashMap<String, f64>) -> Self {
        let mut vocab_to_ids = HashMap::new();
        let mut ids_to_vocab = HashMap::new();
        let mut vocab = vocab;
        vocab.insert("[UNK]".to_string(), 0.0);
        for token in vocab.keys() {
            let id = vocab_to_ids.len() as u32;
            vocab_to_ids.insert(token.clone(), id);
            ids_to_vocab.insert(id, token.clone());
        }
        Tokenizer { vocab, vocab_to_ids, ids_to_vocab, code_pre_tokenizer: CodePreTokenizer::new() }
    }

    fn train(&mut self, corpus: Vec<String>, vocab_size: usize, shrinking_factor: f64) {
        // 1. Initialize with all single characters
        let mut vocab: HashMap<String, f64> = (0..256).map(|i| ((i as u8) as char).to_string()).map(|c| (c, 0.0)).collect();

        for line in &corpus {
            for c in line.chars() {
                vocab.entry(c.to_string()).or_insert(0.0);
            }
        }
        self.vocab = vocab;

        // 2. EM algorithm
        for _ in 0..10 { // EM iterations
            // E-step: Segment the text and count frequencies
            let mut freqs = HashMap::new();
            for line in &corpus {
                let segmentation = self.viterbi_segment(line);
                for (token, _) in segmentation {
                    *freqs.entry(token).or_insert(0) += 1;
                }
            }

            // M-step: Update scores (log probabilities)
            let total: f64 = freqs.values().sum::<i32>() as f64;
            self.vocab = freqs.into_iter().map(|(k, v)| (k, (v as f64 / total).log10())).collect();

            // Prune the vocabulary if it's too large
            if self.vocab.len() > vocab_size {
                let mut sorted_vocab: Vec<_> = self.vocab.iter().collect();
                sorted_vocab.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap());

                let num_to_remove = (self.vocab.len() as f64 * shrinking_factor) as usize;
                let mut to_remove = sorted_vocab.into_iter().rev().take(num_to_remove).map(|(k, _)| k.clone()).collect::<Vec<_>>();

                // Always keep single characters
                to_remove.retain(|t| t.len() > 1);

                for token in to_remove {
                    self.vocab.remove(&token);
                }
            }
        }

        // Update the id mappings
        self.vocab_to_ids.clear();
        self.ids_to_vocab.clear();
        self.vocab.insert("[UNK]".to_string(), 0.0);
        for token in self.vocab.keys() {
            let id = self.vocab_to_ids.len() as u32;
            self.vocab_to_ids.insert(token.clone(), id);
            self.ids_to_vocab.insert(id, token.clone());
        }
    }

    fn encode(&self, text: &str, mode: Option<&str>) -> PyResult<EncodeResult> {
        let pre_tokenized_text = if mode == Some("code") {
            self.code_pre_tokenizer.pre_tokenize(text)
        } else {
            text.to_string()
        };

        let segmentation = self.viterbi_segment(&pre_tokenized_text);
        let mut ids = Vec::new();
        let mut offsets = Vec::new();
        for (token, offset) in segmentation {
            ids.push(*self.vocab_to_ids.get(&token).unwrap_or(&self.vocab_to_ids["[UNK]"]));
            offsets.push(offset);
        }
        Ok((ids, offsets))
    }

    fn decode(&self, ids: Vec<u32>, mode: Option<&str>) -> PyResult<String> {
        let tokens: Vec<String> = ids.iter().filter(|&id| *id != self.vocab_to_ids["[UNK]"]).map(|id| self.ids_to_vocab[id].clone()).collect();
        let text = tokens.join("");
        if mode == Some("code") {
            Ok(self.code_pre_tokenizer.reverse(&text))
        } else {
            Ok(text)
        }
    }

    fn viterbi_segment(&self, text: &str) -> Vec<(String, (usize, usize))> {
        let mut result = Vec::new();
        let mut offset = 0;
        for part in text.split('_') {
            let part_offset = offset;
            let mut chart = vec![None; part.len() + 1];
            chart[0] = Some((0.0, None));

            for (i, _) in part.char_indices().chain(std::iter::once((part.len(), ' '))) {
                if i == 0 { continue; }
                for (j, _) in part[..i].char_indices() {
                    let sub = &part[j..i];
                    if let Some(score) = self.vocab.get(sub) {
                        if let Some(prev) = &chart[j] {
                            let candidate_score = prev.0 + score;
                            if chart[i].is_none() || candidate_score > chart[i].as_ref().unwrap().0 {
                                chart[i] = Some((candidate_score, Some(j)));
                            }
                        }
                    }
                }
            }

            let mut segmentation = Vec::new();
            let mut i = part.len();
            while i > 0 {
                if let Some(entry) = &chart[i] {
                    if let Some(j) = entry.1 {
                        segmentation.push((part[j..i].to_string(), (part_offset + j, part_offset + i)));
                        i = j;
                    } else {
                        break;
                    }
                } else {
                    // Fallback for unknown characters by character
                    if let Some((new_i, last_char)) = part[..i].char_indices().next_back() {
                        segmentation.push((last_char.to_string(), (part_offset + new_i, part_offset + i)));
                        i = new_i;
                    } else {
                        break;
                    }
                }
            }
            segmentation.reverse();
            result.extend(segmentation);
            offset += part.len() + 1; // +1 for the underscore
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_unigram_train_and_encode() {
        let mut tokenizer = Tokenizer::new(HashMap::new());
        let corpus = vec!["hello world".to_string(), "hello".to_string()];
        tokenizer.train(corpus, 10, 0.5);

        let (ids, _) = tokenizer.encode("hello", None).unwrap();
        let tokens: Vec<String> = ids.iter().map(|id| tokenizer.ids_to_vocab[id].clone()).collect();
        // This is a simple test, in reality the segmentation would be more complex
        assert!(tokens.contains(&"he".to_string()) || tokens.contains(&"hel".to_string()));
    }
}
