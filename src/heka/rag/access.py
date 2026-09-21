"""Access control: turn "who is asking" into a metadata filter that retrieval always applies.

The filter is built from the `RequestContext` your application supplies (never from the model or the
question) and is ANDed with every other filter, so nothing downstream can widen it:

    tenant_field: tenant_id      ->  {tenant_id:     {$in: [<ctx.tenant_id>, "*"]}}
    roles_field:  allowed_roles  ->  {allowed_roles: {$in: [*<ctx.roles>,   "*"]}}
    attribute_fields: {region: region}  ->  {region: {$in: [<ctx.attributes.region>, "*"]}}

A chunk without the tag simply fails the filter, so untagged content is invisible (fail-closed).
"""

from __future__ import annotations

from typing import Any

from heka.rag.config import AccessConfig
from heka.rag.errors import AccessDeniedError
from heka.rag.filters import combine
from heka.rag.types import Filter, RequestContext


class AccessPolicy:
    def __init__(self, config: AccessConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def _fields(self) -> dict[str, str]:
        """Chunk metadata field -> what it guards (for messages)."""
        fields: dict[str, str] = {}
        if self.config.tenant_field:
            fields[self.config.tenant_field] = "tenant"
        if self.config.roles_field:
            fields[self.config.roles_field] = "roles"
        for attribute, field in self.config.attribute_fields.items():
            fields[field] = f"attribute '{attribute}'"
        return fields

    def filter_for(self, context: RequestContext | None) -> Filter | None:
        """The filter every retrieval for this caller must satisfy (None when access is not enabled)."""
        if not self.enabled:
            return None
        shared = self.config.shared_value
        if context is None:
            if self.config.require_context:
                raise AccessDeniedError(
                    "An access policy is configured, so every question needs a RequestContext "
                    "(tenant / roles) saying who is asking."
                )
            context = RequestContext()
        parts: list[Filter | None] = []
        if self.config.tenant_field:
            tenants = [context.tenant_id, shared] if context.tenant_id else [shared]
            parts.append({self.config.tenant_field: {"$in": tenants}})
        if self.config.roles_field:
            parts.append({self.config.roles_field: {"$in": [*context.roles, shared]}})
        for attribute, field in self.config.attribute_fields.items():
            value = context.attributes.get(attribute)
            allowed: list[Any] = [value, shared] if value is not None else [shared]
            parts.append({field: {"$in": allowed}})
        return combine(*parts)

    def missing_fields(self, metadata: dict[str, Any]) -> list[str]:
        """Tags this chunk lacks. Such a chunk is invisible to everyone; ingestion warns about it."""
        return [field for field in self._fields() if metadata.get(field) in (None, "", [])]
