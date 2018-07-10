#[macro_use]
extern crate structopt;

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use structopt::StructOpt;

type Result<T> = ::std::result::Result<T, Box<::std::error::Error>>;

#[derive(StructOpt, Debug)]
#[structopt(
    name = "vocabulary", about = "Create the full vocabulary of concepts and their frequency."
)]
struct VocabularyArgs {
    #[structopt(name = "FILE", parse(from_os_str))]
    files: Vec<PathBuf>,

    #[structopt(long = "no-frequency", help = "Only print concept names in order")]
    no_frequency: bool,

    #[structopt(short = "o", long = "output", default_value = "vocabulary.csv", parse(from_os_str))]
    output: PathBuf,
}

#[derive(StructOpt, Debug)]
#[structopt(
    name = "convert",
    about = "Convert the file of lists of concepts into a file of lists of concept ids."
)]
struct ConvertArgs {
    #[structopt(name = "FILE", parse(from_os_str))]
    file: PathBuf,

    #[structopt(long = "no-frequency", help = "Only print concept names in order")]
    no_frequency: bool,

    #[structopt(
        short = "v",
        long = "vocabulary",
        default_value = "training-concepts-ids.csv",
        parse(from_os_str)
    )]
    vocabulary: PathBuf,

    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output: PathBuf,
}

#[derive(StructOpt, Debug)]
#[structopt(name = "concept_count", about = "ImageCLEFcaption concept counting and processing")]
enum ConceptCount {
    #[structopt(
        name = "vocabulary", about = "Create the full vocabulary of concepts and their frequency."
    )]
    Vocabulary(VocabularyArgs),

    #[structopt(
        name = "convert",
        about = "Convert the file of lists of concepts into a file of lists of concept ids."
    )]
    Convert(ConvertArgs),
}

fn main() {
    let args = ConceptCount::from_args();

    match args {
        ConceptCount::Vocabulary(args) => command_vocabulary(&args),
        ConceptCount::Convert(args) => command_convert(&args),
    }.unwrap()
}

fn command_vocabulary(args: &VocabularyArgs) -> Result<()> {
    let paths = &args.files;

    let files = paths
        .into_iter()
        .map(|p| BufReader::new(File::open(p).unwrap()));
    let rows = files.flat_map(|f| f.lines().map(|r| r.unwrap()));

    let concepts = rows.map(|l| {
        let i = l.find("\t").unwrap();
        let split = l.split_at(i);
        (split.0.to_string(), split.1[1..].to_string())
    });

    let concept_lists = concepts.map(|(id, row)| {
        (
            id,
            row.as_str()
                .split(|c| c == ',' || c == ';')
                .filter(|t| !t.is_empty())
                .map(|t| t.to_string())
                .collect::<Vec<_>>(),
        )
    });

    let mut voc_table: HashMap<String, u32> = HashMap::new();
    for (_, concepts) in concept_lists {
        for c in concepts.iter() {
            let counter = voc_table.entry(c.to_string()).or_insert(0);
            *counter += 1;
        }
    }

    let no_frequency = args.no_frequency;
    {
        let mut file = File::create(&args.output)?;
        let mut voc_entries: Vec<_> = voc_table.iter().collect();
        voc_entries
            .as_mut_slice()
            .sort_by(|a, b| a.1.cmp(b.1).reverse());
        for (t, c) in voc_entries {
            if no_frequency {
                writeln!(file, "{}", t)?;
            } else {
                writeln!(file, "{}\t{}", t, c)?;
            }
        }
    }
    println!("Vocabulary saved in {}", args.output.display());
    Ok(())
}

fn command_convert(args: &ConvertArgs) -> Result<()> {
    let voc_file = BufReader::new(File::open(&args.vocabulary)?);
    let vocabulary: HashMap<String, u32> = voc_file
        .lines()
        .map(|l| l.unwrap().split("\t").next().unwrap().to_string())
        .enumerate()
        .map(|(v, k)| (k, v as u32))
        .collect();

    let input = BufReader::new(File::open(&args.file)?);
    let output_path = &args.output;
    let mut output = File::create(&output_path)?;

    let rows = input.lines().map(|r| r.unwrap());

    let concept_id_lists = rows.map(|l| {
        let i = l.find("\t").unwrap();
        let split = l.split_at(i);
        let (id, row) = (split.0.to_string(), split.1[1..].to_string());

        let list = row.as_str()
            .split(',')
            .filter(|t| !t.is_empty())
            .map(|t| t.to_string())
            .filter_map(|c| vocabulary.get(&c).map(|v| *v))
            .collect::<Vec<_>>();

        (id, list)
    });

    for (id, concept_ids) in concept_id_lists {
        write!(output, "{}\t", id)?;
        let mut ids = concept_ids.into_iter();
        if let Some(start) = ids.next() {
            let l: String = ids.fold(start.to_string(), |a, b| {
                a.to_string() + "," + &b.to_string()
            });
            write!(output, "{}", l)?;
        }
        writeln!(output)?;
    }

    println!("Converted concepts saved in {}", output_path.display());
    Ok(())
}
