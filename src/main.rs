use std::{path::Path, time::Instant};

use abd_clam::{Dataset, PartitionCriteria, ShardedCakes, VecDataset};
use distances::Number;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

mod ann_readers;

enum Metrics {
    Cosine,
    Euclidean,
}

impl Metrics {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "cosine" | "angular" => Ok(Metrics::Cosine),
            "euclidean" => Ok(Metrics::Euclidean),
            _ => Err(format!("Unknown metric: {}", s)),
        }
    }

    fn distance(&self) -> fn(&[f32], &[f32]) -> f32 {
        match self {
            Metrics::Cosine => distances::vectors::cosine,
            Metrics::Euclidean => distances::simd::euclidean_f32,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
struct Report<'a> {
    data_name: &'a str,
    metric_name: &'a str,
    cardinality: usize,
    dimensionality: usize,
    shard_sizes: Vec<usize>,
    num_queries: usize,
    k: usize,
    algorithm: &'a str,
    elapsed: Vec<f32>,
}

impl Report<'_> {
    fn save(&self, directory: &Path) -> Result<(), String> {
        let path = directory.join(format!(
            "{}_{}_{}_{}_{}.json",
            self.data_name,
            self.metric_name,
            self.k,
            self.algorithm,
            self.shard_sizes.len()
        ));
        let report = serde_json::to_string(&self).map_err(|e| e.to_string())?;
        std::fs::write(path, report).map_err(|e| e.to_string())?;
        Ok(())
    }
}

fn main() -> Result<(), String> {
    let reports_dir = {
        let mut dir = std::env::current_dir().map_err(|e| e.to_string())?;
        dir.push("reports");
        if !dir.exists() {
            std::fs::create_dir(&dir).map_err(|e| e.to_string())?;
        }
        dir
    };

    for &(data_name, metric_name) in ann_readers::DATASETS {
        if data_name != "sift" {
            continue;
        }

        let metric = if let Ok(metric) = Metrics::from_str(metric_name) {
            metric.distance()
        } else {
            continue;
        };
        println!("dataset: {data_name}, metric: {metric_name}");

        let (train_data, queries) = ann_readers::read_search_data(data_name)?;
        let (cardinality, dimensionality) = (train_data.len(), train_data[0].len());
        println!("cardinality: {cardinality}, dimensionality: {dimensionality}");

        let train_data = train_data.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let queries = queries.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let num_queries = queries.len();
        println!("num_queries: {num_queries}");

        let data = VecDataset::new(data_name.to_string(), train_data, metric, false);
        let threshold = data.cardinality().as_f64().log2().ceil() as usize;
        let criteria = PartitionCriteria::new(true).with_min_cardinality(threshold);
        let cakes = ShardedCakes::new(data, Some(42), criteria, cardinality / 10, 10, 10);

        for k in (0..=3).map(|v| 10usize.pow(v)) {
            println!("\tk: {k}");

            let algorithm = cakes.fastest_algorithm;
            println!("\t\talgorithm: {}", algorithm.name());

            let elapsed = queries
                .par_iter()
                .map(|query| {
                    let start = Instant::now();
                    let hits = cakes.knn_search(query, k);
                    let elapsed = start.elapsed().as_secs_f32();
                    drop(hits);
                    elapsed
                })
                .collect::<Vec<_>>();

            Report {
                data_name,
                metric_name,
                cardinality,
                dimensionality,
                shard_sizes: vec![cardinality],
                num_queries,
                k,
                algorithm: algorithm.name(),
                elapsed,
            }
            .save(&reports_dir)?;
        }

        drop(cakes);
    }

    Ok(())
}
