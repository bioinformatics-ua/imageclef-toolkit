use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;
use structopt::StructOpt;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

#[derive(StructOpt, Debug)]
struct VocabularyArgs {
    /// Concept ground truth file(s)
    #[structopt(name = "FILE", parse(from_os_str))]
    files: Vec<PathBuf>,

    /// Omit frequency column, only print concept names in order
    #[structopt(long = "no-frequency")]
    no_frequency: bool,

    /// Path to the output file
    #[structopt(
        short = "o",
        long = "output",
        default_value = "vocabulary.csv",
        parse(from_os_str)
    )]
    output: PathBuf,
}

#[derive(StructOpt, Debug)]
struct ConvertArgs {
    /// Concept ground truth file(s)
    #[structopt(name = "FILE", parse(from_os_str))]
    file: PathBuf,

    /// Omit frequency column, only print concept names in order
    #[structopt(long = "no-frequency")]
    no_frequency: bool,

    #[structopt(
        short = "v",
        long = "vocabulary",
        default_value = "training-concepts-ids.csv",
        parse(from_os_str)
    )]
    vocabulary: PathBuf,

    /// Path to the output file
    #[structopt(short = "o", long = "output", parse(from_os_str))]
    output: PathBuf,

    /// Delimiter of image IDs in the output file
    #[structopt(
        long = "separator",
        default_value = ","
    )]
    separator: String,
}

#[derive(StructOpt, Debug)]
struct TransposeArgs {
    /// Concept ground truth file(s)
    #[structopt(name = "FILE", parse(from_os_str))]
    file: PathBuf,

    /// Path to the output file
    #[structopt(
        short = "o",
        long = "output",
        default_value = "inverse-map.csv",
        parse(from_os_str)
    )]
    output: PathBuf,

    /// Sort image IDs in each concept
    #[structopt(long = "sort-image-ids")]
    sort_image_ids: bool,

    /// Sort concepts by descending image count
    #[structopt(long = "sort-concepts")]
    sort_concepts: bool,

    /// Add image count as the output's second column
    #[structopt(long = "include-count")]
    include_count: bool,

    /// Delimiter of image IDs in the output file
    #[structopt(
        long = "separator",
        default_value = ","
    )]
    separator: String,
}

/// ImageCLEF Caption concept counting and processing
#[derive(StructOpt, Debug)]
enum ConceptCount {
    #[structopt(
        name = "vocabulary",
        about = "Create the full vocabulary of concepts and their frequency"
    )]
    Vocabulary(VocabularyArgs),

    #[structopt(
        name = "convert",
        about = "Convert the file of lists of concepts into a file of lists of concept IDs"
    )]
    Convert(ConvertArgs),

    #[structopt(
        name = "transpose",
        about = "Create an inverse mapping of concepts to IDs"
    )]
    Transpose(TransposeArgs),
}

fn main() {
    let args = ConceptCount::from_args();

    match args {
        ConceptCount::Vocabulary(args) => command_vocabulary(&args),
        ConceptCount::Convert(args) => command_convert(&args),
        ConceptCount::Transpose(args) => command_transpose(&args),
    }
    .unwrap()
}

fn command_vocabulary(args: &VocabularyArgs) -> Result<()> {
    let paths = &args.files;

    let files = paths
        .into_iter()
        .map(File::open)
        .map(|r| r.map(BufReader::new));
    let rows = files.flat_map(|f| f.unwrap().lines().map(|r| r.unwrap()));

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

        let list = row
            .as_str()
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
                a.to_string() + &args.separator + &b.to_string()
            });
            write!(output, "{}", l)?;
        }
        writeln!(output)?;
    }

    println!("Converted concepts saved in {}", output_path.display());
    Ok(())
}

fn command_transpose(args: &TransposeArgs) -> Result<()> {
    let file_data = ::std::fs::read_to_string(&args.file)?;
    let mut map: HashMap<&str, Vec<&str>> = HashMap::new();

    for l in file_data.lines() {
        let mut cols = l.split("\t");
        let (i, concepts) = (
            cols.next().unwrap(),
            cols.next().unwrap().split(|c| c == ';' || c == ','),
        );
        for c in concepts {
            map.entry(c).or_insert_with(Vec::new).push(i);
        }
    }

    if args.sort_image_ids {
        for v in map.values_mut() {
            v.sort_unstable();
        }
    }

    let mut output = BufWriter::new(File::create(&args.output)?);

    // convert to Vec
    let mut map: Vec<(_, _)> = map.into_iter().collect();
    if args.sort_concepts {
        map.sort_by_key(|(_, e)| usize::max_value() - e.len());
    }

    for (concept_id, image_ids) in map {
        write!(output, "{}\t", concept_id)?;
        if args.include_count {
            write!(output, "{}\t", image_ids.len())?;
        }
        let mut ids = image_ids.into_iter();
        if let Some(start) = ids.next() {
            let l: String = ids.fold(start.to_string(), |a, b| {
                a.to_string() + &args.separator + &b.to_string()
            });
            writeln!(output, "{}", l)?;
        } else {
            writeln!(output)?;
        }
    }

    Ok(())
}
