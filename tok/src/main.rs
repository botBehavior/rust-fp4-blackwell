// Standalone tokenizer helper (normal cargo, NOT cuda-oxide).
//   tok encode "prompt text"   -> space-separated token ids (with BOS)
//   tok decode 1 2 3 ...       -> detokenized text
use tokenizers::Tokenizer;

fn main() {
    let path = "./models/tinyllama/tokenizer.json";
    let tk = Tokenizer::from_file(path).expect("load tokenizer.json");
    let args: Vec<String> = std::env::args().collect();
    match args.get(1).map(|s| s.as_str()) {
        Some("encode") => {
            let text = args[2..].join(" ");
            let enc = tk.encode(text, true).expect("encode");
            let ids: Vec<String> = enc.get_ids().iter().map(|i| i.to_string()).collect();
            println!("{}", ids.join(" "));
        }
        Some("decode") => {
            let ids: Vec<u32> = args[2..].iter().map(|s| s.parse().unwrap()).collect();
            println!("{}", tk.decode(&ids, false).expect("decode"));
        }
        _ => eprintln!("usage: tok encode <text> | tok decode <id...>"),
    }
}
