// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Built-in [`TableReader`] implementations for common data formats.
//!
//! - [`ParquetTableReader`] — reads Parquet tables using a DuckDB-backed
//!   `datafusion-table-providers` integration for local and HTTP paths when
//!   enabled, and falls back to DataFusion's native support otherwise.
//! - [`DeltaTableReader`] — reads Delta Lake tables (behind `delta` feature flag).

use std::collections::HashMap;
use std::sync::Arc;

#[cfg(feature = "duckdb-table-provider")]
use std::sync::OnceLock;

use async_trait::async_trait;
use datafusion::execution::context::SessionContext;
#[cfg(feature = "duckdb-table-provider")]
use datafusion::sql::TableReference;
#[cfg(feature = "duckdb-table-provider")]
use datafusion_table_providers::{
    duckdb::DuckDBTableFactory, sql::db_connection_pool::duckdbpool::DuckDbConnectionPool,
};

use lance_graph_catalog::catalog_provider::{
    CatalogError, CatalogResult, DataSourceFormat, TableInfo,
};
use lance_graph_catalog::table_reader::TableReader;

#[cfg(feature = "duckdb-table-provider")]
static DUCKDB_PARQUET_POOL: OnceLock<Arc<DuckDbConnectionPool>> = OnceLock::new();

async fn register_datafusion_parquet_table(
    ctx: &SessionContext,
    table_name: &str,
    location: &str,
) -> CatalogResult<()> {
    ctx.register_parquet(
        table_name,
        location,
        datafusion::datasource::file_format::options::ParquetReadOptions::default(),
    )
    .await
    .map_err(|e| {
        CatalogError::Other(format!(
            "Failed to register Parquet table '{}' at '{}': {}",
            table_name, location, e
        ))
    })
}

#[cfg(feature = "duckdb-table-provider")]
fn should_use_duckdb_parquet(location: &str, storage_options: &HashMap<String, String>) -> bool {
    if !storage_options.is_empty() {
        return false;
    }

    match url::Url::parse(location) {
        Ok(url) => matches!(url.scheme(), "file" | "http" | "https"),
        Err(_) => !location.contains("://"),
    }
}

#[cfg(feature = "duckdb-table-provider")]
fn duckdb_parquet_pool() -> CatalogResult<Arc<DuckDbConnectionPool>> {
    if let Some(pool) = DUCKDB_PARQUET_POOL.get() {
        return Ok(pool.clone());
    }

    let pool = Arc::new(DuckDbConnectionPool::new_memory().map_err(|e| {
        CatalogError::Other(format!(
            "Failed to initialize DuckDB parquet provider pool: {}",
            e
        ))
    })?);

    let _ = DUCKDB_PARQUET_POOL.set(pool.clone());

    Ok(DUCKDB_PARQUET_POOL.get().cloned().unwrap_or(pool))
}

#[cfg(feature = "duckdb-table-provider")]
fn duckdb_parquet_table_reference(location: &str) -> TableReference {
    TableReference::bare(format!("read_parquet('{}')", location.replace('\'', "''")))
}

#[cfg(feature = "duckdb-table-provider")]
async fn register_duckdb_parquet_table(
    ctx: &SessionContext,
    table_name: &str,
    location: &str,
) -> CatalogResult<()> {
    let table_factory = DuckDBTableFactory::new(duckdb_parquet_pool()?);
    let table_provider = table_factory
        .table_provider(duckdb_parquet_table_reference(location))
        .await
        .map_err(|e| {
            CatalogError::Other(format!(
                "Failed to build DuckDB-backed Parquet provider for '{}' at '{}': {}",
                table_name, location, e
            ))
        })?;

    ctx.register_table(table_name, table_provider).map_err(|e| {
        CatalogError::Other(format!(
            "Failed to register DuckDB-backed Parquet table '{}': {}",
            table_name, e
        ))
    })?;

    Ok(())
}

/// Reads Parquet tables.
///
/// When the `duckdb-table-provider` feature is enabled, local and HTTP Parquet
/// sources are registered through `datafusion-table-providers` so DataFusion
/// can delegate scan pushdown to DuckDB. Cloud/object-store locations continue
/// to use DataFusion's native Parquet registration path.
pub struct ParquetTableReader;

#[async_trait]
impl TableReader for ParquetTableReader {
    fn name(&self) -> &str {
        "parquet"
    }

    fn supported_formats(&self) -> &[DataSourceFormat] {
        &[DataSourceFormat::Parquet]
    }

    async fn register_table(
        &self,
        ctx: &SessionContext,
        table_name: &str,
        table_info: &TableInfo,
        _schema: arrow_schema::SchemaRef,
        storage_options: &HashMap<String, String>,
    ) -> CatalogResult<()> {
        let location = table_info.storage_location.as_deref().ok_or_else(|| {
            CatalogError::Other(format!("Table '{}' has no storage_location", table_name))
        })?;

        #[cfg(feature = "duckdb-table-provider")]
        if should_use_duckdb_parquet(location, storage_options) {
            return register_duckdb_parquet_table(ctx, table_name, location).await;
        }

        #[cfg(not(feature = "duckdb-table-provider"))]
        let _ = storage_options;

        register_datafusion_parquet_table(ctx, table_name, location).await
    }
}

/// Reads Delta Lake tables using the `deltalake` crate.
///
/// Opens the Delta table at the storage location and registers it as a
/// DataFusion `TableProvider`, enabling full SQL query support including
/// time travel, schema evolution, and partition pruning.
///
/// Supports cloud storage (S3, Azure, GCS) via `storage_options`.
#[cfg(feature = "delta")]
pub struct DeltaTableReader;

#[cfg(feature = "delta")]
#[async_trait]
impl TableReader for DeltaTableReader {
    fn name(&self) -> &str {
        "delta"
    }

    fn supported_formats(&self) -> &[DataSourceFormat] {
        &[DataSourceFormat::Delta]
    }

    async fn register_table(
        &self,
        ctx: &SessionContext,
        table_name: &str,
        table_info: &TableInfo,
        _schema: arrow_schema::SchemaRef,
        storage_options: &HashMap<String, String>,
    ) -> CatalogResult<()> {
        let location = table_info.storage_location.as_deref().ok_or_else(|| {
            CatalogError::Other(format!("Table '{}' has no storage_location", table_name))
        })?;

        let table_url = url::Url::parse(location).map_err(|e| {
            CatalogError::Other(format!(
                "Invalid storage location URL '{}': {}",
                location, e
            ))
        })?;

        let delta_table = if storage_options.is_empty() {
            deltalake::open_table(table_url).await
        } else {
            deltalake::open_table_with_storage_options(table_url, storage_options.clone()).await
        }
        .map_err(|e| {
            CatalogError::Other(format!(
                "Failed to open Delta table '{}' at '{}': {}",
                table_name, location, e
            ))
        })?;

        let table_provider = delta_table.table_provider().await.map_err(|e| {
            CatalogError::Other(format!(
                "Failed to create Delta TableProvider for '{}': {}",
                table_name, e
            ))
        })?;

        ctx.register_table(table_name, table_provider)
            .map_err(|e| {
                CatalogError::Other(format!(
                    "Failed to register Delta table '{}': {}",
                    table_name, e
                ))
            })?;

        Ok(())
    }
}

/// Returns the default set of table readers.
///
/// Includes Parquet support, and Delta Lake support when the `delta` feature is enabled.
pub fn default_table_readers() -> Vec<Arc<dyn TableReader>> {
    let mut readers: Vec<Arc<dyn TableReader>> = vec![Arc::new(ParquetTableReader)];
    #[cfg(feature = "delta")]
    readers.push(Arc::new(DeltaTableReader));
    readers
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fs::File;
    use std::path::Path;

    use arrow::util::pretty::pretty_format_batches;
    use arrow_array::{Int64Array, RecordBatch, StringArray};
    use arrow_schema::{DataType, Field, Schema};
    use lance_graph_catalog::catalog_provider::{ColumnInfo, TableType};
    use parquet::arrow::ArrowWriter;
    use tempfile::tempdir;

    fn write_test_parquet(path: &Path) -> arrow_schema::SchemaRef {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("name", DataType::Utf8, false),
        ]));

        let batch = RecordBatch::try_new(
            schema.clone(),
            vec![
                Arc::new(Int64Array::from(vec![1, 2, 3])),
                Arc::new(StringArray::from(vec!["alice", "bob", "carol"])),
            ],
        )
        .expect("test parquet batch");

        let file = File::create(path).expect("create parquet file");
        let mut writer = ArrowWriter::try_new(file, schema.clone(), None).expect("parquet writer");
        writer.write(&batch).expect("write parquet batch");
        writer.close().expect("close parquet writer");

        schema
    }

    fn parquet_table_info(location: String) -> TableInfo {
        TableInfo {
            name: "people".to_string(),
            catalog_name: "test".to_string(),
            schema_name: "default".to_string(),
            table_type: TableType::External,
            data_source_format: DataSourceFormat::Parquet,
            columns: vec![
                ColumnInfo {
                    name: "id".to_string(),
                    type_text: "BIGINT".to_string(),
                    type_name: "BIGINT".to_string(),
                    position: 0,
                    nullable: false,
                    comment: None,
                },
                ColumnInfo {
                    name: "name".to_string(),
                    type_text: "STRING".to_string(),
                    type_name: "STRING".to_string(),
                    position: 1,
                    nullable: false,
                    comment: None,
                },
            ],
            storage_location: Some(location),
            comment: None,
            properties: HashMap::new(),
            created_at: None,
            updated_at: None,
        }
    }

    #[cfg(feature = "duckdb-table-provider")]
    #[test]
    fn duckdb_parquet_selection_is_conservative() {
        assert!(should_use_duckdb_parquet("/tmp/test.parquet", &HashMap::new()));
        assert!(should_use_duckdb_parquet(
            "https://example.com/test.parquet",
            &HashMap::new()
        ));
        assert!(!should_use_duckdb_parquet(
            "s3://bucket/test.parquet",
            &HashMap::new()
        ));

        let mut storage_options = HashMap::new();
        storage_options.insert("aws_access_key_id".to_string(), "key".to_string());
        assert!(!should_use_duckdb_parquet(
            "/tmp/test.parquet",
            &storage_options
        ));
    }

    #[cfg(feature = "duckdb-table-provider")]
    #[tokio::test]
    async fn parquet_reader_registers_duckdb_backed_provider_for_local_files() {
        let temp_dir = tempdir().expect("temp dir");
        let parquet_path = temp_dir.path().join("people.parquet");
        let schema = write_test_parquet(&parquet_path);

        let ctx = SessionContext::new_with_state(datafusion_federation::default_session_state());
        let reader = ParquetTableReader;
        let table_info = parquet_table_info(parquet_path.to_string_lossy().into_owned());

        reader
            .register_table(&ctx, "people", &table_info, schema, &HashMap::new())
            .await
            .expect("register parquet table");

        let result = ctx
            .sql("SELECT name FROM people WHERE id = 2")
            .await
            .expect("query dataframe")
            .collect()
            .await
            .expect("query result");
        let formatted = pretty_format_batches(&result)
            .expect("format result")
            .to_string();
        assert!(formatted.contains("bob"), "unexpected query result: {formatted}");
    }
}
