"""Search, question answering, summarisation and classification endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Query, status
from fastapi.responses import JSONResponse

from idengine.api.deps import Admin, EngineDep, Reader
from idengine.api.schemas import (
    AnswerOut,
    AskRequest,
    ClassifyRequest,
    JobOut,
    PredictionOut,
    SearchRequest,
    SearchResponseOut,
    SummarizeTextRequest,
    SummaryOut,
    TrainingResultOut,
    TrainRequest,
)
from idengine.engine import LabeledExample

router = APIRouter(prefix="/v1", tags=["intelligence"])


@router.post("/search", response_model=SearchResponseOut)
def search(
    engine: EngineDep,
    _: Reader,
    request: SearchRequest,
    include_text: Annotated[bool, Query(description="Return full chunk text")] = False,
) -> SearchResponseOut:
    response = engine.search(
        request.query,
        mode=request.mode,
        top_k=request.top_k,
        filters=request.filters,
        group_by_document=request.group_by_document,
    )
    return SearchResponseOut.from_response(response, include_text=include_text)


@router.post("/ask", response_model=AnswerOut)
def ask(engine: EngineDep, _: Reader, request: AskRequest) -> AnswerOut:
    answer = engine.ask(
        request.question, filters=request.filters, top_k=request.top_k, mode=request.mode
    )
    return AnswerOut.from_answer(answer)


@router.post("/summarize", response_model=SummaryOut)
def summarize_text(engine: EngineDep, _: Reader, request: SummarizeTextRequest) -> SummaryOut:
    summary = engine.summarize_text(
        request.text, method=request.method, max_sentences=request.max_sentences
    )
    return SummaryOut.from_summary(summary)


@router.get("/classifier")
def classifier_status(engine: EngineDep, _: Reader) -> dict[str, object]:
    return engine.classifier.status()


@router.post(
    "/classifier/train",
    response_model=TrainingResultOut | JobOut,
    responses={202: {"description": "Training job queued"}},
)
def train_classifier(
    engine: EngineDep, _: Admin, request: TrainRequest
) -> TrainingResultOut | JSONResponse:
    examples = [
        LabeledExample(label=e.label, text=e.text, document_id=e.document_id)
        for e in request.examples
    ]
    if request.background:
        job = engine.jobs.submit(
            "train_classifier",
            {
                "examples": [
                    {"label": e.label, "text": e.text, "document_id": e.document_id}
                    for e in examples
                ],
                "algorithm": request.algorithm,
                "test_ratio": request.test_ratio,
                "folds": request.folds,
            },
        )
        return JSONResponse(
            JobOut.from_record(job).model_dump(mode="json"), status_code=status.HTTP_202_ACCEPTED
        )
    result = engine.train_classifier(
        examples, algorithm=request.algorithm, test_ratio=request.test_ratio, folds=request.folds
    )
    return TrainingResultOut.from_result(result)


@router.post("/classify", response_model=PredictionOut)
def classify_text(engine: EngineDep, _: Reader, request: ClassifyRequest) -> PredictionOut:
    return PredictionOut.from_prediction(engine.classify_text(request.text))
