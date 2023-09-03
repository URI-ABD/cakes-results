mod ann_readers;
mod ann_reports;
mod bigann_readers;

fn main() -> Result<(), String> {
    ann_reports::make_reports()?;
    Ok(())
}
