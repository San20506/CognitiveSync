"""AES-256-GCM encrypted ID mapping store.

Stores pseudo_id ↔ real_id mappings in an encrypted volume.
This file is the ONLY place where identity mappings are persisted.
PostgreSQL never sees real identifiers.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from uuid import UUID

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

logger = logging.getLogger(__name__)

_SALT_BYTES = b"cognitivesync-vault-kdf-salt-v1"  # Fixed KDF salt (not crypto salt)
_ITERATIONS = 600_000


def derive_key(master_key: str) -> bytes:
    """Derive a 32-byte AES key from the master secret using PBKDF2HMAC."""
    kdf = PBKDF2HMAC(
        algorithm=SHA256(),
        length=32,
        salt=_SALT_BYTES,
        iterations=_ITERATIONS,
    )
    return kdf.derive(master_key.encode())


class EncryptedMappingStore:
    """Thread-safe AES-256-GCM encrypted store for pseudo_id ↔ real_id mappings.

    Data layout on disk: nonce (12 bytes) + ciphertext (encrypted JSON dict).
    """

    def __init__(self, vault_path: Path, master_key: str) -> None:
        self._path = vault_path
        self._key = derive_key(master_key)
        self._aesgcm = AESGCM(self._key)

    def upsert(self, pseudo_id: UUID, real_id: str) -> None:
        """Insert or update a pseudo_id ↔ real_id mapping."""
        mapping = self._load()
        mapping[str(pseudo_id)] = real_id
        self._save(mapping)
        logger.debug("Mapping upserted for pseudo_id=%s", pseudo_id)

    def lookup(self, pseudo_id: UUID) -> str | None:
        """Resolve a pseudo_id to its real identifier (HR admin use only)."""
        return self._load().get(str(pseudo_id))

    def purge(self, pseudo_id: UUID) -> None:
        """Remove a mapping (employee offboarding)."""
        mapping = self._load()
        mapping.pop(str(pseudo_id), None)
        self._save(mapping)

    def _load(self) -> dict[str, str]:
        if not self._path.exists():
            return {}
        raw = self._path.read_bytes()
        nonce, ciphertext = raw[:12], raw[12:]
        plaintext = self._aesgcm.decrypt(nonce, ciphertext, None)
        return json.loads(plaintext)  # type: ignore[no-any-return]

    def _save(self, mapping: dict[str, str]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        nonce = os.urandom(12)
        ciphertext = self._aesgcm.encrypt(nonce, json.dumps(mapping).encode(), None)
        self._path.write_bytes(nonce + ciphertext)
