"""Jobs, health, stats, metrics and admin endpoints."""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import APIRouter, Path, Query, Request, Response, status
from fastapi.responses import JSONResponse

from idengine.api.deps import Admin, EngineDep, Reader, Writer
from idengine.api.schemas import JobList, JobOut, ReindexRequest
from idengine.api.security import Authenticator, authorize
from idengine.config import Role

router = APIRouter(tags=["system"])

JobId = Annotated[str, Path(pattern=r"^job_[0-9a-f]{32}$")]


@router.get("/health")
def health(engine: EngineDep) -> JSONResponse:
    report = engine.health()
    code = status.HTTP_200_OK if report["status"] == "ok" else status.HTTP_503_SERVICE_UNAVAILABLE
    return JSONResponse(report, status_code=code)


@router.get("/v1/stats")
def stats(engine: EngineDep, _: Reader) -> dict[str, Any]:
    return engine.stats()


@router.get("/metrics", response_class=Response)
def metrics(engine: EngineDep, request: Request) -> Response:
    if not engine.settings.api.metrics_public:
        authenticator: Authenticator = request.app.state.authenticator
        authorize(authenticator.authenticate(request), Role.READER)
    engine.stats()  # refresh gauges
    return Response(engine.metrics.render(), media_type="text/plain; version=0.0.4")


@router.get("/v1/jobs", response_model=JobList, tags=["jobs"])
def list_jobs(
    engine: EngineDep,
    _: Reader,
    status_: Annotated[
        Literal["queued", "running", "succeeded", "failed", "cancelled"] | None,
        Query(alias="status"),
    ] = None,
    job_type: Annotated[str | None, Query(alias="type", max_length=40)] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> JobList:
    jobs, total = engine.jobs.repository.list(
        status=status_, job_type=job_type, limit=limit, offset=offset
    )
    return JobList(
        items=[JobOut.from_record(j) for j in jobs], total=total, limit=limit, offset=offset
    )


@router.get("/v1/jobs/{job_id}", response_model=JobOut, tags=["jobs"])
def get_job(engine: EngineDep, _: Reader, job_id: JobId) -> JobOut:
    return JobOut.from_record(engine.jobs.repository.get(job_id))


@router.post("/v1/jobs/{job_id}/cancel", response_model=JobOut, tags=["jobs"])
def cancel_job(engine: EngineDep, _: Writer, job_id: JobId) -> JobOut:
    return JobOut.from_record(engine.jobs.repository.cancel(job_id))


@router.post("/v1/jobs/{job_id}/retry", response_model=JobOut, tags=["jobs"])
def retry_job(engine: EngineDep, _: Admin, job_id: JobId) -> JobOut:
    return JobOut.from_record(engine.jobs.repository.retry(job_id))


@router.post("/v1/admin/reindex", tags=["admin"])
def reindex(engine: EngineDep, _: Admin, request: ReindexRequest) -> JSONResponse:
    if request.background:
        job = engine.jobs.submit("reindex", {"refit": request.refit})
        return JSONResponse(
            JobOut.from_record(job).model_dump(mode="json"), status_code=status.HTTP_202_ACCEPTED
        )
    return JSONResponse(engine.reindex(refit=request.refit))
