use std::time::Instant;

use abd_clam::{knn::Algorithm, Cakes, PartitionCriteria, VecDataset};

mod ann_readers;

enum Metrics {
    Cosine,
    Euclidean,
}

impl Metrics {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "cosine" => Ok(Metrics::Cosine),
            "euclidean" => Ok(Metrics::Euclidean),
            _ => Err(format!("Unknown metric: {}", s)),
        }
    }

    fn distance(&self) -> fn(&[f32], &[f32]) -> f32 {
        match self {
            Metrics::Cosine => distances::vectors::cosine,
            Metrics::Euclidean => distances::vectors::euclidean,
        }
    }
}

fn main() -> Result<(), String> {
    for &(data_name, metric_name) in ann_readers::DATASETS {
        if data_name != "mnist" {
            continue;
        }

        let metric = if let Ok(metric) = Metrics::from_str(metric_name) {
            metric.distance()
        } else {
            continue;
        };
        println!("dataset: {data_name}, metric: {metric_name}");

        let (train_data, queries) = ann_readers::read_search_data(data_name)?;
        println!("cardinality: {}, dimensionality: {}", train_data.len(), train_data[0].len());
        
        let train_data = train_data.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let train_data = VecDataset::new(data_name.to_string(), train_data, metric, false);

        let criteria = PartitionCriteria::default();
        let cakes = Cakes::new(train_data, None, criteria);

        let queries = queries.iter().take(10_000).map(Vec::as_slice).collect::<Vec<_>>();
        let num_queries = queries.len();
        println!("num_queries: {num_queries}");

        for k in (0..3).map(|v| 10usize.pow(v)) {
            println!("\tk: {k}");

            for &algorithm in Algorithm::variants() {
                println!("\t\talgorithm: {}", algorithm.name());

                let start = Instant::now();
                let results = cakes.batch_knn_search(&queries, k, algorithm);
                let elapsed = start.elapsed().as_secs_f32();
                let throughput = num_queries as f32 / elapsed;

                println!("\t\ttime = {elapsed:.3e} sec, throughput: {throughput:.3e} QPS");
                drop(results);
            }
        }
    }

    Ok(())
}
