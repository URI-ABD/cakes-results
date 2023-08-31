use core::cmp::Ordering;
use std::{path::Path, time::Instant};

use abd_clam::{knn, rnn, Cakes, Dataset, PartitionCriteria, ShardedCakes, VecDataset};
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
        if data_name != "glove-25" {
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

        let data = VecDataset::new(data_name.to_string(), train_data.clone(), metric, false);
        let threshold = data.cardinality().as_f64().log2().ceil() as usize;
        let criteria = PartitionCriteria::new(true).with_min_cardinality(threshold);
        let cakes = Cakes::new(data, Some(42), criteria);

        let queries = queries.iter().map(Vec::as_slice).collect::<Vec<_>>();
        let num_queries = queries.len();
        println!("num_queries: {num_queries}");

        let mut ks_radii = vec![];
        for k in (0..=10).map(|v| 2usize.pow(v)) {
            println!("\tk: {k}");

            let mut radii = vec![];
            for &algorithm in knn::Algorithm::variants() {
                println!("\t\talgorithm: {}", algorithm.name());

                let (radii_, elapsed): (Vec<_>, Vec<_>) = queries
                    .par_iter()
                    .map(|query| {
                        let start = Instant::now();
                        let knn_hits = cakes.knn_search(query, k, algorithm);
                        let elapsed = start.elapsed().as_secs_f32();

                        let radius = knn_hits
                            .into_iter()
                            .map(|(_, d)| d)
                            .max_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Less))
                            .unwrap();
                        (radius, elapsed)
                    })
                    .unzip();

                radii = radii_;

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

            for &algorithm in rnn::Algorithm::variants() {
                if matches!(algorithm, rnn::Algorithm::Linear) {
                    continue;
                }

                println!("\t\talgorithm: {}", algorithm.name());

                let elapsed = queries
                    .par_iter()
                    .zip(radii.par_iter())
                    .map(|(&query, &radius)| {
                        let start = Instant::now();
                        let hits = cakes.rnn_search(query, radius, algorithm);
                        let elapsed = start.elapsed().as_secs_f32();
                        drop(hits);
                        elapsed
                    })
                    .collect();

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

            ks_radii.push((k, radii));
        }

        drop(cakes);

        // Run benchmarks on multiple shards
        for num_shards in (1..8).map(|v| 2usize.pow(v)) {
            let shards = train_data
                .chunks(cardinality / num_shards)
                .enumerate()
                .map(|(i, data)| {
                    let name = format!("shard-{}", i);
                    VecDataset::new(name, data.to_vec(), metric, false)
                })
                .map(|s| {
                    let threshold = s.cardinality().as_f64().log2().ceil() as usize;
                    let criteria = PartitionCriteria::new(true).with_min_cardinality(threshold);
                    (s, criteria)
                })
                .collect::<Vec<_>>();
            let shard_sizes = shards
                .iter()
                .map(|(s, _)| s.cardinality())
                .collect::<Vec<_>>();
            println!("num_shards: {}", shard_sizes.len());

            let sharded_cakes = ShardedCakes::new(shards, Some(42));

            for (k, radii) in ks_radii.iter().map(|(k, radii)| (*k, radii)) {
                println!("\tk: {k}");

                for &algorithm in knn::Algorithm::variants() {
                    println!("\t\talgorithm: {}", algorithm.name());

                    let elapsed = queries
                        .par_iter()
                        .map(|query| {
                            let start = Instant::now();
                            let hits = sharded_cakes.knn_search(query, k, algorithm);
                            let elapsed = start.elapsed().as_secs_f32();
                            drop(hits);
                            elapsed
                        })
                        .collect();

                    Report {
                        data_name,
                        metric_name,
                        cardinality,
                        dimensionality,
                        shard_sizes: vec![1],
                        num_queries,
                        k,
                        algorithm: algorithm.name(),
                        elapsed,
                    }
                    .save(&reports_dir)?;
                }

                for &algorithm in rnn::Algorithm::variants() {
                    if matches!(algorithm, rnn::Algorithm::Linear) {
                        continue;
                    }

                    println!("\t\talgorithm: {}", algorithm.name());

                    let elapsed = queries
                        .par_iter()
                        .zip(radii.par_iter())
                        .map(|(&query, &radius)| {
                            let start = Instant::now();
                            let hits = sharded_cakes.rnn_search(query, radius, algorithm);
                            let elapsed = start.elapsed().as_secs_f32();
                            drop(hits);
                            elapsed
                        })
                        .collect();

                    Report {
                        data_name,
                        metric_name,
                        cardinality,
                        dimensionality,
                        shard_sizes: shard_sizes.clone(),
                        num_queries,
                        k,
                        algorithm: algorithm.name(),
                        elapsed,
                    }
                    .save(&reports_dir)?;
                }
            }
        }
    }

    Ok(())
}
