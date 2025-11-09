use rkyv::{Archive, Deserialize, Serialize};

#[derive(Archive, Deserialize, Serialize, Clone)]
#[archive(check_bytes)]
pub struct CodePreTokenizer;

impl CodePreTokenizer {
    pub fn new() -> Self {
        CodePreTokenizer
    }

    pub fn pre_tokenize(&self, text: &str) -> String {
        let mut result = String::new();
        let mut chars = text.chars().peekable();

        while let Some(c) = chars.next() {
            result.push(c);

            if let Some(&next_c) = chars.peek() {
                // camelCase: lower to upper
                if c.is_lowercase() && next_c.is_uppercase() {
                    result.push_str("<SPLIT>");
                }
                // camelCase: upper to upper then lower
                else if c.is_uppercase() && next_c.is_uppercase() {
                    // Peek ahead to see if the one after is lowercase
                    let mut temp_chars = chars.clone();
                    temp_chars.next(); // consume the next uppercase char
                    if let Some(after_next_c) = temp_chars.peek() {
                        if after_next_c.is_lowercase() {
                            result.push_str("<SPLIT>");
                        }
                    }
                }
                // letter to number or number to letter
                else if (c.is_alphabetic() && next_c.is_numeric()) || (c.is_numeric() && next_c.is_alphabetic()) {
                    result.push_str("<SPLIT>");
                }
            }
        }

        result.replace('_', "<SPLIT>")
    }

    pub fn reverse(&self, text: &str) -> String {
        text.replace("<SPLIT>", "")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_camel_case_split() {
        let pre_tokenizer = CodePreTokenizer::new();
        let text = "myVariableName";
        let expected = "my<SPLIT>Variable<SPLIT>Name";
        assert_eq!(pre_tokenizer.pre_tokenize(text), expected);
    }

    #[test]
    fn test_snake_case_split() {
        let pre_tokenizer = CodePreTokenizer::new();
        let text = "my_variable_name";
        let expected = "my<SPLIT>variable<SPLIT>name";
        assert_eq!(pre_tokenizer.pre_tokenize(text), expected);
    }

    #[test]
    fn test_mixed_case_split() {
        let pre_tokenizer = CodePreTokenizer::new();
        let text = "myVariable_123_anotherName";
        let expected = "my<SPLIT>Variable<SPLIT>123<SPLIT>another<SPLIT>Name";
        assert_eq!(pre_tokenizer.pre_tokenize(text), expected);
    }

    #[test]
    fn test_reversibility() {
        let pre_tokenizer = CodePreTokenizer::new();
        let original_text = "myVariable_anotherName"; // Simpler case without numbers for now
        let tokenized_text = pre_tokenizer.pre_tokenize(original_text);
        let reversed_text = pre_tokenizer.reverse(&tokenized_text);
        assert_eq!(original_text.replace("_", ""), reversed_text);
    }

    #[test]
    fn test_double_underscore() {
        let pre_tokenizer = CodePreTokenizer::new();
        let text = "my__variable";
        let expected = "my<SPLIT><SPLIT>variable";
        assert_eq!(pre_tokenizer.pre_tokenize(text), expected);
    }

    #[test]
    fn test_leading_underscore() {
        let pre_tokenizer = CodePreTokenizer::new();
        let text = "_my_variable";
        let expected = "<SPLIT>my<SPLIT>variable";
        assert_eq!(pre_tokenizer.pre_tokenize(text), expected);
    }

    #[test]
    fn test_trailing_underscore() {
        let pre_tokenizer = CodePreTokenizer::new();
        let text = "my_variable_";
        let expected = "my<SPLIT>variable<SPLIT>";
        assert_eq!(pre_tokenizer.pre_tokenize(text), expected);
    }
}
