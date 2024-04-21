use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub(crate) struct Args {
    #[command(subcommand)]
    pub(crate) cmd: Command,
}

#[derive(Parser, Debug)]
pub(crate) enum Command {
    Encode(Encode),
    Decode(Decode),
    /// Test is a combination of both encode and decode
    /// It takes the same arguments as encode
    Test(Encode),
}

#[derive(Parser, Debug, Clone)]
pub(crate) struct Encode {
    pub(crate) file: String,

    #[arg(short, long, default_value_t = 6)]
    pub(crate) max_depth: usize,

    #[arg(short, long, default_value_t = 5.0)]
    pub(crate) detail_threshold: f32,

    #[arg(short, long)]
    pub(crate) output: Option<String>,
}

#[derive(Parser, Debug)]
pub(crate) struct Decode {
    pub(crate) file: String,

    #[arg(short, long)]
    pub(crate) output: Option<String>,
}
