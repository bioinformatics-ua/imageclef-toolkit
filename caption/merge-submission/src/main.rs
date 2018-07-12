#[macro_use]
extern crate quicli;

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use quicli::prelude::*;

#[derive(Debug, StructOpt)]
#[structopt(
    name = "merge-submission",
    about = "Merge multiple ImageCLEF 2018 concept detection submission files into one"
)]
struct Cli {
    #[structopt(help = "The submission files to merge")]
    #[structopt(raw(required = "true"))]
    file: Vec<String>,
    #[structopt(short = "d", long = "delimiter", default_value = ";", help = "The concept delimiter")]
    delimiter: String,
}

main!(|args: Cli| {
    run_strict(&args.file, &args.delimiter)
});

fn run_strict<P: AsRef<Path>>(files: &[P], delimiter: &str) {
    let files = files
        .iter()
        .map(|f| File::open(f).unwrap())
        .map(BufReader::new);

    let mut file_lines = files.map(|f| f.lines()).map(|lines| {
        lines.map(|l| {
            let l = l.unwrap();
            let mut parts = l.split('\t');
            let fid = parts.next().unwrap().to_string();
            let concepts = parts.next().unwrap().to_string();
            (fid, concepts)
        })
    });

    let mut o: Vec<(String, String)> = file_lines.next().unwrap().collect();

    while let Some(lines) = file_lines.next() {
        for (&mut (ref fid1, ref mut concepts1), (ref fid2, ref concepts2)) in
            o.iter_mut().zip(lines)
        {
            assert_eq!(fid1, fid2, "Identifiers are not in the same order.");
            if concepts2.len() > 0 {
                concepts1.push_str(delimiter);
                concepts1.push_str(&concepts2);
            }
        }
    }

    for (fid, concepts) in o {
        println!("{}\t{}", fid, concepts);
    }
}
