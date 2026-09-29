//! The serving surface: Arrow Flight over an embedded online store.
//!
//! # Why the transport is here and not in Python
//!
//! This crate is the whole request path. A client's request is decoded here, the store is read
//! here, values are decoded and TTL-checked here, and the response is built and serialized here.
//! Python imports this module and blocks; it is the host of the server, not a participant in it.
//!
//! That is what lets `serve()` be a method on a user's `FeatureStore` and still use every core.
//! If the handler were Python, the GIL would serialise every request, the answer would be process
//! workers, and more processes is exactly what an embedded store cannot be opened by. Measured:
//! a Python host serves at 0.82 to 0.92 of a Rust one, and the GIL is provably released. See
//! `docs/serving-transport.md`.
//!
//! # What a request is
//!
//! A `DoGet` ticket carries a [`ReadRequest`]: a feature service name and the entity keys to read
//! for it. The field set is **not** in the ticket. It is resolved once at startup by
//! [`ResolvedService::resolve`] and held here, so a request cannot make the serving path resolve
//! metadata per call. That is item 3 of issue #5.
//!
//! # The read
//!
//! [`read_entities`] does the work, which is the point: the read-time TTL check lives there and
//! is the authority on whether a value is served. This crate does not read the store directly,
//! because a store read without that check would serve expired values.
//!
//! One store read covers the whole request however many entities and views it names, which is
//! the rule the collocated one-hash-per-entity layout exists to make possible.
//!
//! The response is one `RecordBatch` per entity, streamed as a single Flight response. One
//! response rather than N round trips is the point of the transport; per-entity batches rather
//! than one concatenated batch is what avoids needing a concat, and the schema is identical
//! across them.
//!
//! `ponytail:` the ticket is JSON rather than protobuf. It is a few microseconds of parsing on a
//! path whose floor is the store read, and a hand-written encoder is not worth maintaining.
//! Upgrade path: an Arrow-encoded descriptor, which is a change to one function.

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::net::SocketAddr;
use std::sync::Arc;

use arrow::array::{ArrayRef, new_null_array};
use arrow::datatypes::{Field, Fields, Schema, SchemaRef};
use arrow::ipc::writer::IpcWriteOptions;
use arrow::record_batch::RecordBatch;
use arrow_flight::encode::FlightDataEncoderBuilder;
use arrow_flight::flight_service_server::{FlightService, FlightServiceServer};
use arrow_flight::{
    Action, ActionType, Criteria, Empty, FlightData, FlightDescriptor, FlightInfo,
    HandshakeRequest, HandshakeResponse, PollInfo, PutResult, SchemaAsIpc, SchemaResult, Ticket,
};
use futures_util::stream::{self, StreamExt};
use tonic::transport::Server;
use tonic::{Request, Response, Status};

use feather_core::arrow_type;
use feather_core::definitions::FeatureView;
use feather_core::online::{EntityRequest, ViewRequest, ViewValues, read_entities};

/// The store, behind a pointer so every worker shares one.
pub type SharedStore = Arc<feather_core::online::fjall::FjallStore>;

/// The body of a `DoGet` ticket.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ReadRequest {
    /// The feature service being read. Checked against what this server resolved, so a client
    /// cannot ask for a projection the server did not load at startup.
    pub service: String,
    /// One encoded entity key per row to return, as the caller produced it.
    pub entities: Vec<Vec<u8>>,
}

/// What a client declares about a feature service, before it is resolved.
#[derive(Debug, Clone)]
pub struct ServiceSpec {
    pub name: String,
    /// The entity every view in the service shares, which is what makes one hash key per entity
    /// possible: the views are collocated, so one store read covers all of them.
    pub entity_name: String,
    /// `(view, feature)` per requested column, **in the order the client wants them back**.
    pub features: Vec<(String, String)>,
}

/// One output column: which view it came from, which of that view's returned columns it is,
/// and the field it appears as.
struct OutputColumn {
    /// Index into the per-entity `ViewValues` list, which is grouped by view.
    view_at: usize,
    /// Index into that view's returned columns, which is the order the template asked for.
    field_at: usize,
    field: Field,
}

/// A feature service resolved against the definitions, once, at startup.
pub struct ResolvedService {
    pub project: String,
    pub name: String,
    /// The views the service references, for `read_entities` to resolve names against.
    views: BTreeMap<String, FeatureView>,
    /// The per-entity request, built once: which views, and which of their features.
    template: Vec<ViewRequest>,
    /// One entry per output column, in the order the service declared them.
    columns: Vec<OutputColumn>,
    schema: SchemaRef,
}

impl ResolvedService {
    /// Resolve a declared service against the project's views.
    ///
    /// This is the "resolve once at startup" of issue #5, and it is where a bad reference is
    /// caught: a service naming a view or a feature that does not exist fails here, at boot,
    /// rather than on the first request.
    pub fn resolve(
        project: &str,
        spec: &ServiceSpec,
        views: &BTreeMap<String, FeatureView>,
    ) -> Result<Self, String> {
        if spec.features.is_empty() {
            return Err(format!("feature service `{}` names no features", spec.name));
        }

        // The store read is grouped by view because a view's whole encoded vector is one field
        // and one read covers all the features asked of it. The output order is the service's
        // declared order, so a column records where in that grouped list it came from.
        let mut template: Vec<ViewRequest> = Vec::new();
        let mut view_at: HashMap<&str, usize> = HashMap::new();
        let mut columns: Vec<OutputColumn> = Vec::with_capacity(spec.features.len());

        for (view_name, feature_name) in &spec.features {
            let view = views.get(view_name).ok_or_else(|| {
                format!(
                    "feature service `{}` names unknown view `{view_name}`",
                    spec.name
                )
            })?;
            let at = *view_at.entry(view_name.as_str()).or_insert_with(|| {
                template.push(ViewRequest {
                    view: view_name.clone(),
                    fields: Vec::new(),
                });
                template.len() - 1
            });
            let index = template[at]
                .fields
                .iter()
                .position(|f| f == feature_name)
                .unwrap_or_else(|| {
                    template[at].fields.push(feature_name.clone());
                    template[at].fields.len() - 1
                });
            let declared = view
                .features
                .iter()
                .find(|f| &f.name == feature_name)
                .ok_or_else(|| {
                    format!(
                        "feature service `{}` names `{feature_name}`, which view `{view_name}` does not declare",
                        spec.name
                    )
                })?;
            columns.push(OutputColumn {
                view_at: at,
                field_at: index,
                // Qualified, because two views may declare a feature of the same name and a bare
                // name would be ambiguous in a response.
                field: Field::new(
                    format!("{view_name}:{feature_name}"),
                    arrow_type(declared.dtype),
                    true,
                ),
            });
        }

        let schema = Arc::new(Schema::new(Fields::from_iter(
            columns.iter().map(|c| c.field.clone()),
        )));
        Ok(Self {
            project: project.to_owned(),
            name: spec.name.clone(),
            views: views.clone(),
            template,
            columns,
            schema,
        })
    }

    /// The schema every batch in a response carries.
    pub fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

type BoxStream<T> =
    std::pin::Pin<Box<dyn futures_util::Stream<Item = Result<T, Status>> + Send + 'static>>;

fn empty<T: Send + 'static>() -> BoxStream<T> {
    stream::empty().boxed()
}

/// The IPC form of the response schema, which is what `GetSchema` hands back.
fn schema_ipc(schema: &Schema) -> Result<SchemaResult, Status> {
    SchemaAsIpc::new(schema, &IpcWriteOptions::default())
        .try_into()
        .map_err(|e| Status::internal(format!("could not encode the response schema: {e}")))
}

/// Wall clock in microseconds, which is the unit the read-time TTL check takes.
fn now_unix_micros() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_micros() as i64)
        .unwrap_or_default()
}

/// The Flight service. Cheap to clone-share through an `Arc`; the store is behind one.
pub struct FeatureFlightService {
    store: SharedStore,
    service: ResolvedService,
}

impl FeatureFlightService {
    pub fn new(store: SharedStore, service: ResolvedService) -> Self {
        Self { store, service }
    }

    /// The real read: one store call for the request, TTL-checked, decoded, one batch per entity.
    async fn read_entities(&self, read: &ReadRequest) -> Result<Vec<RecordBatch>, Status> {
        let entities: Vec<EntityRequest> = read
            .entities
            .iter()
            .map(|encoded_key| EntityRequest {
                encoded_key: encoded_key.clone(),
                views: self.service.template.clone(),
            })
            .collect();

        let rows = read_entities(
            self.store.as_ref(),
            &self.service.project,
            &self.service.views,
            &entities,
            now_unix_micros(),
        )
        .await
        .map_err(|e| Status::internal(format!("online read failed: {e}")))?;

        let mut batches = Vec::with_capacity(rows.len());
        for per_view in &rows {
            let mut arrays: Vec<ArrayRef> = Vec::with_capacity(self.service.columns.len());
            for column in &self.service.columns {
                let value = match per_view.get(column.view_at) {
                    // Missing is a null of the declared type, not a dropped column: the entity
                    // was asked for, and a caller has to be able to see that a feature is absent,
                    // which is how a TTL expiry arrives.
                    Some(ViewValues::Present { columns, .. }) => columns
                        .get(column.field_at)
                        .map(Arc::clone)
                        .unwrap_or_else(|| new_null_array(&column.field.data_type().clone(), 1)),
                    _ => new_null_array(&column.field.data_type().clone(), 1),
                };
                arrays.push(value);
            }
            batches.push(
                RecordBatch::try_new(Arc::clone(&self.service.schema), arrays)
                    .map_err(|e| Status::internal(format!("bad response batch: {e}")))?,
            );
        }
        Ok(batches)
    }
}

#[tonic::async_trait]
impl FlightService for FeatureFlightService {
    type HandshakeStream = BoxStream<HandshakeResponse>;
    type ListFlightsStream = BoxStream<FlightInfo>;
    type DoGetStream = BoxStream<FlightData>;
    type DoPutStream = BoxStream<PutResult>;
    type DoExchangeStream = BoxStream<FlightData>;
    type DoActionStream = BoxStream<arrow_flight::Result>;
    type ListActionsStream = BoxStream<ActionType>;

    async fn handshake(
        &self,
        _request: Request<tonic::Streaming<HandshakeRequest>>,
    ) -> Result<Response<Self::HandshakeStream>, Status> {
        // No authentication yet, and refusing is the honest answer rather than accepting
        // everything. Issue #5 leaves this open and the Flight specification is explicit that a
        // token validated only at connection time is not safe behind a layer-7 load balancer, so
        // the answer is per-call validation or mTLS. Until then, loopback is the only thing
        // protecting this endpoint, which is why `serve` binds there by default.
        Err(Status::unimplemented(
            "feather serve does not authenticate yet",
        ))
    }

    async fn list_flights(
        &self,
        _request: Request<Criteria>,
    ) -> Result<Response<Self::ListFlightsStream>, Status> {
        Ok(Response::new(empty()))
    }

    async fn get_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<FlightInfo>, Status> {
        let info = FlightInfo::new()
            .try_with_schema(self.service.schema().as_ref())
            .map_err(|e| Status::internal(e.to_string()))?;
        Ok(Response::new(info))
    }

    async fn poll_flight_info(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<PollInfo>, Status> {
        // Nothing is pending: a DoGet is served from the store as it arrives rather than from a
        // queue behind a producer, so there is no flight for a client to poll.
        Ok(Response::new(PollInfo {
            info: None,
            progress: Some(0.0),
            ..Default::default()
        }))
    }

    async fn get_schema(
        &self,
        _request: Request<FlightDescriptor>,
    ) -> Result<Response<SchemaResult>, Status> {
        Ok(Response::new(schema_ipc(self.service.schema().as_ref())?))
    }

    async fn do_get(
        &self,
        request: Request<Ticket>,
    ) -> Result<Response<Self::DoGetStream>, Status> {
        let read: ReadRequest = serde_json::from_slice(request.get_ref().ticket.as_ref())
            .map_err(|e| Status::invalid_argument(format!("malformed ticket: {e}")))?;

        if read.service != self.service.name {
            return Err(Status::invalid_argument(format!(
                "unknown feature service `{}`; this server serves `{}`",
                read.service, self.service.name
            )));
        }
        if read.entities.is_empty() {
            return Err(Status::invalid_argument("no entities in the ticket"));
        }

        let batches = self.read_entities(&read).await?;
        let schema = self.service.schema();
        // The encoder's stream yields `arrow_flight::Result`, whose error is an encode failure
        // rather than a gRPC one, so it is translated rather than passed through.
        let encoded = FlightDataEncoderBuilder::new()
            .with_schema(schema)
            .build(stream::iter(batches.into_iter().map(Ok)))
            .map(|item| item.map_err(|e| Status::internal(e.to_string())))
            .boxed();
        Ok(Response::new(encoded))
    }

    async fn do_put(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoPutStream>, Status> {
        Err(Status::unimplemented(
            "feather serve does not accept writes over DoPut",
        ))
    }

    async fn do_exchange(
        &self,
        _request: Request<tonic::Streaming<FlightData>>,
    ) -> Result<Response<Self::DoExchangeStream>, Status> {
        Err(Status::unimplemented("feather serve has no DoExchange"))
    }

    async fn do_action(
        &self,
        _request: Request<Action>,
    ) -> Result<Response<Self::DoActionStream>, Status> {
        Err(Status::unimplemented("feather serve has no actions"))
    }

    async fn list_actions(
        &self,
        _request: Request<Empty>,
    ) -> Result<Response<Self::ListActionsStream>, Status> {
        Ok(Response::new(empty()))
    }
}

/// Serve until `shutdown` resolves.
///
/// Blocking a thread on this is the intended use: `FeatureStore.serve()` in Python hands the
/// current thread to it and releases the GIL, which is what keeps the server's own workers free
/// to run and the caller's other threads alive. A shutdown future that is already ready, such as
/// `async {}`, stops the server at once, which looks exactly like one that never bound.
pub async fn serve<F>(
    addr: SocketAddr,
    store: SharedStore,
    service: ResolvedService,
    shutdown: F,
) -> Result<(), tonic::transport::Error>
where
    F: Future<Output = ()> + Send + 'static,
{
    let schema = service.schema();
    let svc = FeatureFlightService::new(store, service);
    tracing::info!(
        "feather serve listening on {addr} serving {} columns",
        schema.fields().len()
    );
    Server::builder()
        .add_service(FlightServiceServer::new(svc))
        .serve_with_shutdown(addr, shutdown)
        .await
}
