use omnitoken_core::Tokenizer;
use pyo3::prelude::*;
use rkyv::{Archive, Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[pyclass]
#[derive(Archive, Deserialize, Serialize)]
#[archive(check_bytes)]
pub struct Model {
    #[pyo3(get, set)]
    pub tokenizer: Tokenizer,
}

#[pymethods]
impl Model {
    #[new]
    pub fn new(tokenizer: Tokenizer) -> Self {
        Model { tokenizer }
    }

    pub fn save(&self, path: &str) -> PyResult<()> {
        let bytes = rkyv::to_bytes::<_, 256>(self).unwrap();
        let mut file = File::create(path)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    #[staticmethod]
    pub fn load(path: &str) -> PyResult<Self> {
        let mut file = File::open(path)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        let archived = rkyv::check_archived_root::<Self>(&bytes[..]).unwrap();
        let model = archived.deserialize(&mut rkyv::Infallible).unwrap();
        Ok(model)
    }
}

#[pymodule]
fn _native(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<Model>()?;
    m.add_class::<Tokenizer>()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_save_and_load() {
        let mut vocab = HashMap::new();
        vocab.insert("a".to_string(), 0.1);
        vocab.insert("b".to_string(), 0.2);
        let tokenizer = Tokenizer::new(vocab);
        let model = Model::new(tokenizer);

        let path = "test_model.bin";
        model.save(path).unwrap();

        let loaded_model = Model::load(path).unwrap();

        assert_eq!(model.tokenizer.vocab, loaded_model.tokenizer.vocab);

        std::fs::remove_file(path).unwrap();
    }
}
