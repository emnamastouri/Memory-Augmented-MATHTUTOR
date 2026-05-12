"""Gestion simple des comptes avec MongoDB local."""

from __future__ import annotations

import base64
from datetime import UTC, datetime, timedelta
import hashlib
import hmac
import os
import re
import secrets
from typing import Any
import unicodedata

import streamlit as st
from pymongo import ASCENDING, MongoClient
from pymongo.collection import Collection
from pymongo.errors import DuplicateKeyError, PyMongoError

from frontend.utils.constants import (
    AUTH_ROLES,
    LEVELS,
    MONGO_DB_NAME,
    MONGO_EXERCISE_RECORDS_COLLECTION,
    MONGO_LEARNING_EVENTS_COLLECTION,
    MONGO_TUTORING_THREADS_COLLECTION,
    MONGO_URI,
    MONGO_USERS_COLLECTION,
)
from frontend.utils.dataset_catalog import normalize_section_label

EMAIL_PATTERN = re.compile(r"^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$")
PASSWORD_RESET_TTL_MINUTES = 15
RESET_TOKEN_LENGTH = 8


@st.cache_resource(show_spinner=False)
def get_mongo_client() -> MongoClient:
    """Creer un client Mongo reutilisable pour l'application."""
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=3000)


def get_users_collection() -> Collection:
    """Retourner la collection des utilisateurs et garantir l'unicite de l'email."""
    collection = get_mongo_client()[MONGO_DB_NAME][MONGO_USERS_COLLECTION]
    collection.create_index([("email", ASCENDING)], unique=True)
    return collection


def register_user(
    *,
    name: str,
    email: str,
    password: str,
    role: str,
    level: str,
    section: str,
) -> dict[str, Any]:
    """Creer un compte dans MongoDB."""
    validation = validate_registration_data(
        name=name,
        email=email,
        password=password,
        confirm_password=password,
        role=role,
        level=level,
        section=section,
    )
    if not validation["ok"]:
        return {
            "ok": False,
            "message": validation["errors"][0],
            "errors": validation["errors"],
            "password_strength": validation["password_strength"],
        }

    password_payload = _hash_password(password)
    user_document = {
        "name": validation["clean_name"],
        "email": validation["normalized_email"],
        "password_hash": password_payload["hash"],
        "password_salt": password_payload["salt"],
        "role": validation["normalized_role"],
        "level": validation["normalized_level"],
        "section": validation["normalized_section"],
        "created_at": datetime.now(UTC),
        "password_updated_at": datetime.now(UTC),
    }

    try:
        result = get_users_collection().insert_one(user_document)
    except DuplicateKeyError:
        return {"ok": False, "message": "Un compte existe deja avec cette adresse e-mail."}
    except PyMongoError as exc:
        return {"ok": False, "message": f"Connexion MongoDB indisponible : {exc}"}

    user_document["_id"] = result.inserted_id
    return {"ok": True, "message": "Compte cree avec succes.", "user": _serialize_user(user_document)}


def login_user(email: str, password: str) -> dict[str, Any]:
    """Authentifier un utilisateur enregistre dans MongoDB."""
    normalized_email = email.strip().lower()
    try:
        user_document = get_users_collection().find_one({"email": normalized_email})
    except PyMongoError as exc:
        return {"ok": False, "message": f"Connexion MongoDB indisponible : {exc}"}

    if not user_document:
        return {"ok": False, "message": "Aucun compte trouve pour cette adresse e-mail."}

    if not _verify_password(password, user_document["password_salt"], user_document["password_hash"]):
        return {"ok": False, "message": "Mot de passe incorrect."}

    return {"ok": True, "message": "Connexion reussie.", "user": _serialize_user(user_document)}


def get_user_account(email: str) -> dict[str, Any]:
    """Charger les informations du compte courant depuis MongoDB."""
    normalized_email = email.strip().lower()
    if not normalized_email:
        return {"ok": False, "message": "Adresse e-mail manquante."}

    try:
        user_document = get_users_collection().find_one({"email": normalized_email})
    except PyMongoError as exc:
        return {"ok": False, "message": f"Connexion MongoDB indisponible : {exc}"}

    if not user_document:
        return {"ok": False, "message": "Compte introuvable."}

    account = _serialize_user(user_document)
    account["created_at"] = _coerce_utc_datetime(user_document.get("created_at"))
    account["password_updated_at"] = _coerce_utc_datetime(user_document.get("password_updated_at"))
    account["profile_updated_at"] = _coerce_utc_datetime(user_document.get("profile_updated_at"))
    return {"ok": True, "account": account}


def update_user_profile(
    *,
    email: str,
    name: str,
    section: str = "",
) -> dict[str, Any]:
    """Mettre a jour les donnees personnelles modifiables du compte."""
    normalized_email = email.strip().lower()
    if not normalized_email:
        return {"ok": False, "message": "Adresse e-mail manquante."}

    try:
        user_document = get_users_collection().find_one({"email": normalized_email})
    except PyMongoError as exc:
        return {"ok": False, "message": f"Connexion MongoDB indisponible : {exc}"}

    if not user_document:
        return {"ok": False, "message": "Compte introuvable."}

    clean_name = _clean_name(name)
    normalized_role = normalize_role(str(user_document.get("role", AUTH_ROLES[0])))
    normalized_level = "" if normalized_role == "Enseignant" else normalize_level(str(user_document.get("level", LEVELS[0])))
    normalized_section = "" if normalized_role == "Enseignant" else normalize_section_label(section)

    errors: list[str] = []
    if len(clean_name) < 5:
        errors.append("Le nom complet doit contenir au moins 5 caracteres.")
    if len(clean_name.split()) < 2:
        errors.append("Veuillez saisir un nom et un prenom.")
    if normalized_role != "Enseignant" and not normalized_section:
        errors.append("Veuillez choisir une section.")
    if errors:
        return {"ok": False, "message": errors[0], "errors": errors}

    update_payload = {
        "name": clean_name,
        "level": normalized_level,
        "section": normalized_section,
        "profile_updated_at": datetime.now(UTC),
    }

    try:
        get_users_collection().update_one({"_id": user_document["_id"]}, {"$set": update_payload})
        _sync_account_metadata(
            normalized_email,
            display_name=clean_name,
            level=normalized_level,
            section=normalized_section,
        )
        refreshed_document = get_users_collection().find_one({"_id": user_document["_id"]})
    except PyMongoError as exc:
        return {"ok": False, "message": f"Impossible de mettre a jour le profil : {exc}"}

    if not refreshed_document:
        return {"ok": False, "message": "Le compte n'a pas pu etre relu apres mise a jour."}
    return {"ok": True, "message": "Les informations personnelles ont ete mises a jour.", "user": _serialize_user(refreshed_document)}


def change_user_password(
    *,
    email: str,
    current_password: str,
    new_password: str,
    confirm_password: str,
) -> dict[str, Any]:
    """Modifier le mot de passe du compte connecte apres verification."""
    normalized_email = email.strip().lower()
    if not normalized_email:
        return {"ok": False, "message": "Adresse e-mail manquante."}
    if not current_password or not new_password or not confirm_password:
        return {"ok": False, "message": "Veuillez remplir les trois champs de mot de passe."}
    if new_password != confirm_password:
        return {"ok": False, "message": "La confirmation du mot de passe ne correspond pas."}

    try:
        user_document = get_users_collection().find_one({"email": normalized_email})
    except PyMongoError as exc:
        return {"ok": False, "message": f"Connexion MongoDB indisponible : {exc}"}

    if not user_document:
        return {"ok": False, "message": "Compte introuvable."}
    if not _verify_password(current_password, user_document["password_salt"], user_document["password_hash"]):
        return {"ok": False, "message": "Le mot de passe actuel est incorrect."}
    if current_password == new_password:
        return {"ok": False, "message": "Le nouveau mot de passe doit etre different de l'ancien."}

    strength = evaluate_password_strength(
        new_password,
        email=normalized_email,
        name=user_document.get("name", ""),
    )
    if not strength["ok"]:
        return {
            "ok": False,
            "message": "Le nouveau mot de passe n'atteint pas le niveau de securite requis.",
            "errors": [item["label"] for item in strength["requirements"] if not item["ok"]],
            "password_strength": strength,
        }

    password_payload = _hash_password(new_password)
    try:
        get_users_collection().update_one(
            {"_id": user_document["_id"]},
            {
                "$set": {
                    "password_hash": password_payload["hash"],
                    "password_salt": password_payload["salt"],
                    "password_updated_at": datetime.now(UTC),
                }
            },
        )
    except PyMongoError as exc:
        return {"ok": False, "message": f"Impossible de modifier le mot de passe : {exc}"}

    return {"ok": True, "message": "Le mot de passe a ete mis a jour avec succes."}


def request_password_reset(email: str, name: str) -> dict[str, Any]:
    """Generer un code local de reinitialisation pour un compte existant."""
    normalized_email = email.strip().lower()
    clean_name = _clean_name(name)

    if not normalized_email or not clean_name:
        return {"ok": False, "message": "Veuillez saisir l'adresse e-mail et le nom complet du compte."}
    if not EMAIL_PATTERN.match(normalized_email):
        return {"ok": False, "message": "L'adresse e-mail saisie n'est pas valide."}

    generic_message = (
        "Si un compte correspondant existe, un code temporaire de reinitialisation a ete prepare "
        "pour cette session locale."
    )

    try:
        user_document = get_users_collection().find_one({"email": normalized_email})
    except PyMongoError as exc:
        return {"ok": False, "message": f"Connexion MongoDB indisponible : {exc}"}

    if not user_document or not _names_match(clean_name, user_document.get("name", "")):
        return {"ok": True, "message": generic_message}

    reset_token = _generate_reset_token()
    expires_at = datetime.now(UTC) + timedelta(minutes=PASSWORD_RESET_TTL_MINUTES)

    try:
        get_users_collection().update_one(
            {"_id": user_document["_id"]},
            {
                "$set": {
                    "reset_token_hash": _hash_reset_token(reset_token),
                    "reset_token_expires_at": expires_at,
                    "password_reset_requested_at": datetime.now(UTC),
                }
            },
        )
    except PyMongoError as exc:
        return {"ok": False, "message": f"Impossible de preparer la reinitialisation : {exc}"}

    return {
        "ok": True,
        "message": generic_message,
        "dev_token": reset_token,
        "expires_in_minutes": PASSWORD_RESET_TTL_MINUTES,
    }


def reset_password(
    *,
    email: str,
    reset_token: str,
    new_password: str,
    confirm_password: str,
) -> dict[str, Any]:
    """Finaliser la reinitialisation du mot de passe avec un code temporaire."""
    normalized_email = email.strip().lower()
    provided_token = reset_token.strip().upper()

    if not normalized_email or not provided_token or not new_password or not confirm_password:
        return {"ok": False, "message": "Veuillez remplir l'adresse e-mail, le code et les deux champs de mot de passe."}
    if not EMAIL_PATTERN.match(normalized_email):
        return {"ok": False, "message": "L'adresse e-mail saisie n'est pas valide."}
    if new_password != confirm_password:
        return {"ok": False, "message": "La confirmation du mot de passe ne correspond pas."}

    try:
        user_document = get_users_collection().find_one({"email": normalized_email})
    except PyMongoError as exc:
        return {"ok": False, "message": f"Connexion MongoDB indisponible : {exc}"}

    if not user_document:
        return {"ok": False, "message": "Demande de reinitialisation invalide ou expiree."}

    strength = evaluate_password_strength(
        new_password,
        email=normalized_email,
        name=user_document.get("name", ""),
    )
    if not strength["ok"]:
        return {
            "ok": False,
            "message": "Le nouveau mot de passe n'atteint pas le niveau de securite requis.",
            "errors": [item["label"] for item in strength["requirements"] if not item["ok"]],
            "password_strength": strength,
        }

    stored_hash = user_document.get("reset_token_hash")
    expires_at = user_document.get("reset_token_expires_at")
    expires_at_utc = _coerce_utc_datetime(expires_at)
    if not stored_hash or not expires_at_utc or datetime.now(UTC) > expires_at_utc:
        return {"ok": False, "message": "Le code de reinitialisation a expire. Veuillez en demander un nouveau."}
    if _hash_reset_token(provided_token) != stored_hash:
        return {"ok": False, "message": "Le code de reinitialisation est invalide."}

    password_payload = _hash_password(new_password)
    try:
        get_users_collection().update_one(
            {"_id": user_document["_id"]},
            {
                "$set": {
                    "password_hash": password_payload["hash"],
                    "password_salt": password_payload["salt"],
                    "password_updated_at": datetime.now(UTC),
                    "reset_completed_at": datetime.now(UTC),
                },
                "$unset": {
                    "reset_token_hash": "",
                    "reset_token_expires_at": "",
                    "password_reset_requested_at": "",
                },
            },
        )
    except PyMongoError as exc:
        return {"ok": False, "message": f"Impossible de mettre a jour le mot de passe : {exc}"}

    return {"ok": True, "message": "Le mot de passe a ete reinitialise. Vous pouvez maintenant vous connecter."}


def evaluate_password_strength(password: str, *, email: str = "", name: str = "") -> dict[str, Any]:
    """Evaluer la solidite d'un mot de passe pour l'interface et le backend."""
    similar_tokens = _extract_identity_tokens(email=email, name=name)
    lowered_password = password.lower()
    requirements = [
        {"label": "Au moins 10 caracteres", "ok": len(password) >= 10},
        {"label": "Au moins une lettre minuscule", "ok": any(char.islower() for char in password)},
        {"label": "Au moins une lettre majuscule", "ok": any(char.isupper() for char in password)},
        {"label": "Au moins un chiffre", "ok": any(char.isdigit() for char in password)},
        {"label": "Au moins un caractere special", "ok": any(not char.isalnum() and not char.isspace() for char in password)},
        {"label": "Aucun espace", "ok": password == password.strip() and " " not in password},
        {
            "label": "Different du nom et de l'e-mail",
            "ok": bool(password) and all(token not in lowered_password for token in similar_tokens),
        },
    ]
    score = sum(1 for item in requirements if item["ok"])

    if score <= 2:
        label = "Faible"
    elif score <= 4:
        label = "Moyen"
    elif score <= 6:
        label = "Fort"
    else:
        label = "Tres fort"

    return {
        "ok": all(item["ok"] for item in requirements),
        "score": score,
        "max_score": len(requirements),
        "label": label,
        "requirements": requirements,
    }


def validate_registration_data(
    *,
    name: str,
    email: str,
    password: str,
    confirm_password: str,
    role: str,
    level: str,
    section: str,
) -> dict[str, Any]:
    """Valider les donnees de creation de compte avant insertion MongoDB."""
    clean_name = _clean_name(name)
    normalized_email = email.strip().lower()
    normalized_role = normalize_role(role)
    is_teacher_registration = normalized_role == "Enseignant"
    normalized_level = "" if is_teacher_registration else normalize_level(level)
    normalized_section = "" if is_teacher_registration else normalize_section_label(section)
    strength = evaluate_password_strength(password, email=normalized_email, name=clean_name)

    errors: list[str] = []
    if len(clean_name) < 5:
        errors.append("Le nom complet doit contenir au moins 5 caracteres.")
    if len(clean_name.split()) < 2:
        errors.append("Veuillez saisir un nom et un prenom.")
    if not normalized_email:
        errors.append("L'adresse e-mail est obligatoire.")
    elif not EMAIL_PATTERN.match(normalized_email):
        errors.append("Veuillez saisir une adresse e-mail valide.")
    if normalized_role not in AUTH_ROLES:
        errors.append("Le type de compte selectionne est invalide.")
    if not is_teacher_registration and normalized_level not in LEVELS:
        errors.append("Seul le niveau Bac est actuellement disponible.")
    if not is_teacher_registration and not normalized_section:
        errors.append("Veuillez choisir une section.")
    if not password:
        errors.append("Le mot de passe est obligatoire.")
    if password != confirm_password:
        errors.append("La confirmation du mot de passe ne correspond pas.")
    if not strength["ok"]:
        errors.append("Le mot de passe doit respecter toutes les exigences de securite affichees.")

    return {
        "ok": not errors,
        "errors": errors,
        "clean_name": clean_name,
        "normalized_email": normalized_email,
        "normalized_role": normalized_role,
        "normalized_level": normalized_level,
        "normalized_section": normalized_section,
        "password_strength": strength,
    }


def _hash_password(password: str) -> dict[str, str]:
    """Calculer un hash PBKDF2 pour le mot de passe."""
    salt = os.urandom(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return {
        "salt": base64.b64encode(salt).decode("ascii"),
        "hash": base64.b64encode(digest).decode("ascii"),
    }


def _verify_password(password: str, salt_b64: str, digest_b64: str) -> bool:
    """Comparer le mot de passe fourni au hash enregistre."""
    salt = base64.b64decode(salt_b64.encode("ascii"))
    expected_digest = base64.b64decode(digest_b64.encode("ascii"))
    candidate_digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 120000)
    return hmac.compare_digest(candidate_digest, expected_digest)


def _serialize_user(user_document: dict[str, Any]) -> dict[str, Any]:
    """Preparer un document utilisateur pour la session Streamlit."""
    return {
        "user_id": str(user_document["_id"]),
        "name": user_document["name"],
        "email": user_document["email"],
        "role": user_document["role"],
        "level": user_document.get("level", ""),
        "section": user_document.get("section", ""),
    }


def _sync_account_metadata(email: str, *, display_name: str, level: str, section: str) -> None:
    """Propager les metadonnees utilisateur utiles dans les collections liees."""
    database = get_mongo_client()[MONGO_DB_NAME]
    update_payload = {
        "user_display_name": display_name,
        "level": level,
        "section": section,
    }
    try:
        database[MONGO_LEARNING_EVENTS_COLLECTION].update_many(
            {"user_email": email},
            {"$set": update_payload},
        )
        database[MONGO_EXERCISE_RECORDS_COLLECTION].update_many(
            {"user_email": email},
            {"$set": update_payload},
        )
        database[MONGO_TUTORING_THREADS_COLLECTION].update_many(
            {"user_email": email},
            {"$set": update_payload},
        )
    except PyMongoError:
        return


def _clean_name(name: str) -> str:
    """Nettoyer un nom tout en conservant la casse d'affichage."""
    return " ".join(name.split()).strip()


def _extract_identity_tokens(*, email: str, name: str) -> list[str]:
    """Extraire des fragments a ne pas reutiliser dans le mot de passe."""
    tokens: list[str] = []
    if email:
        email_local = email.split("@", 1)[0].lower()
        if len(email_local) >= 3:
            tokens.append(email_local)
    for part in re.findall(r"[A-Za-zÀ-ÿ0-9]+", name):
        lowered = _normalize_text(part)
        if len(lowered) >= 3:
            tokens.append(lowered)
    return list(dict.fromkeys(tokens))


def normalize_role(role: str) -> str:
    """Normaliser un role meme si l'encodage ou l'accentuation varient."""
    cleaned = role.strip()
    aliases = {_normalize_key(item): item for item in AUTH_ROLES}
    return aliases.get(_normalize_key(cleaned), cleaned)


def normalize_level(level: str) -> str:
    """Normaliser le niveau d'etude."""
    cleaned = level.strip()
    aliases = {_normalize_key(item): item for item in LEVELS}
    return aliases.get(_normalize_key(cleaned), cleaned)


def _generate_reset_token() -> str:
    """Generer un code court pour la reinitialisation locale."""
    alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
    return "".join(secrets.choice(alphabet) for _ in range(RESET_TOKEN_LENGTH))


def _hash_reset_token(reset_token: str) -> str:
    """Hasher un code de reinitialisation temporaire."""
    return hashlib.sha256(reset_token.encode("utf-8")).hexdigest()


def _names_match(provided_name: str, stored_name: str) -> bool:
    """Comparer deux noms en ignorant les accents et les espaces multiples."""
    return _normalize_text(provided_name) == _normalize_text(stored_name)


def _normalize_text(value: str) -> str:
    """Normaliser une chaine pour des comparaisons de securite simples."""
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_value.lower().split())


def _normalize_key(value: str) -> str:
    """Normaliser une valeur courte pour les verifications d'appartenance."""
    return _normalize_text(value)


def _coerce_utc_datetime(value: datetime | None) -> datetime | None:
    """Rattacher un datetime naive a UTC pour les comparaisons avec MongoDB."""
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
