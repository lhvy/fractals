use clap::Parser;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub(crate) struct Args {
    #[command(subcommand)]
    pub(crate) cmd: Command,
}

#[derive(Parser, Debug)]
pub(crate) enum Command {
    Encode(Value),
    Decode(Value),
    Test(Value),
}

#[derive(Parser, Debug, Clone)]
pub(crate) struct Value {
    pub(crate) file: String,

    #[arg(short, long, default_value_t = 4)]
    pub(crate) dest_size: usize,

    #[arg(short, long)]
    pub(crate) output: Option<String>,
}
