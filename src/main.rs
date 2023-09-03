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
    recalls: Vec<f64>,
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
        if !["glove-25", "glove-100"].contains(&data_name) {
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

            let (elapsed, recalls): (Vec<_>, Vec<_>) = queries
                .par_iter()
                .map(|query| {
                    let start = Instant::now();
                    let mut hits = cakes.knn_search(query, k);
                    let elapsed = start.elapsed().as_secs_f32();
                    hits.sort_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater)
                    });

                    let mut linear_hits = cakes.linear_knn(query, k);
                    linear_hits.sort_by(|(_, a), (_, b)| {
                        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater)
                    });

                    let mut hits = hits.into_iter().map(|(_, d)| d).peekable();
                    let mut linear_hits = linear_hits.into_iter().map(|(_, d)| d).peekable();

                    let mut num_common = 0;
                    while let (Some(&hit), Some(&linear_hit)) = (hits.peek(), linear_hits.peek()) {
                        if hit < linear_hit {
                            hits.next();
                        } else if hit > linear_hit {
                            linear_hits.next();
                        } else {
                            num_common += 1;
                            hits.next();
                            linear_hits.next();
                        }
                    }
                    let recall = num_common.as_f64() / k.as_f64();

                    (elapsed, recall)
                })
                .unzip();

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
                recalls,
            }
            .save(&reports_dir)?;
        }

        drop(cakes);
    }

    Ok(())
}
