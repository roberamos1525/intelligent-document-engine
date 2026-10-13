"""FastAPI dependencies: engine access and role-checked principals."""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated

from fastapi import Depends, Request

from idengine.api.security import Authenticator, Principal, authorize
from idengine.config import Role
from idengine.engine import Engine


def get_engine(request: Request) -> Engine:
    engine: Engine = request.app.state.engine
    return engine


def _principal(request: Request) -> Principal:
    authenticator: Authenticator = request.app.state.authenticator
    principal = authenticator.authenticate(request)
    request.state.principal = principal
    return principal


def require(role: Role) -> Callable[[Request], Principal]:
    def dependency(principal: Annotated[Principal, Depends(_principal)]) -> Principal:
        authorize(principal, role)
        return principal

    return dependency


EngineDep = Annotated[Engine, Depends(get_engine)]
Reader = Annotated[Principal, Depends(require(Role.READER))]
Writer = Annotated[Principal, Depends(require(Role.WRITER))]
Admin = Annotated[Principal, Depends(require(Role.ADMIN))]
