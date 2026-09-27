"""API Key storage and resolution hierarchy for multi-provider agent runtime."""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union


class KeyStore:
    """Manages API key resolution across override parameters, environment variables,

    user config files (~/.config/valstorm/keys.json), and workspace .env.ai.keys.

    Precedence order:
    1. override_key (if explicitly provided)
    2. Environment variables (e.g., GEMINI_API_KEY, GOOGLE_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, DEEPSEEK_API_KEY)
    3. User config JSON file (~/.config/valstorm/keys.json)
    4. Repo-level .env.ai.keys file
    """

    PROVIDER_ENV_MAP: Dict[str, List[str]] = {
        "gemini": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "google": ["GEMINI_API_KEY", "GOOGLE_API_KEY"],
        "digitalocean": [
            "DIGITALOCEAN_AI_KEY",
            "DO_GENAI_KEY",
            "DO_INFERENCE_TOKEN",
            "DIGITALOCEAN_ACCESS_TOKEN",
            "DO_API_KEY",
            "DIGITALOCEAN_API_KEY",
            "DO_TOKEN",
        ],
        "do": [
            "DIGITALOCEAN_AI_KEY",
            "DO_GENAI_KEY",
            "DO_INFERENCE_TOKEN",
            "DIGITALOCEAN_ACCESS_TOKEN",
            "DO_API_KEY",
            "DIGITALOCEAN_API_KEY",
            "DO_TOKEN",
        ],
        "openai": ["OPENAI_API_KEY"],
        "anthropic": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "claude": ["ANTHROPIC_API_KEY", "CLAUDE_API_KEY"],
        "deepseek": ["DEEPSEEK_API_KEY"],
        "groq": ["GROQ_API_KEY"],
        "moonshot": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
        "kimi": ["MOONSHOT_API_KEY", "KIMI_API_KEY"],
        "mistral": ["MISTRAL_API_KEY", "CODESTRAL_API_KEY"],
        "codestral": ["CODESTRAL_API_KEY", "MISTRAL_API_KEY"],
        "together": ["TOGETHER_API_KEY"],
        "openrouter": ["OPENROUTER_API_KEY"],
        "siliconflow": ["SILICONFLOW_API_KEY"],
        "ollama": ["OLLAMA_API_KEY"],
        "vllm": ["VLLM_API_KEY"],
        "xai": ["XAI_API_KEY", "GROK_API_KEY"],
        "valstorm": [
            "VALSTORM_API_TOKEN",
            "VALSTORM_JWT_TOKEN",
            "VALSTORM_API_KEY",
            "VALSTORM_TOKEN",
            "VALSTORM_PAT",
            "VALSTORM_PAT_TOKEN",
            "PAT_TOKEN",
            "PAT",
            "VALSTORM_ACCESS_TOKEN",
            "ACCESS_TOKEN",
        ],
    }

    def __init__(
        self,
        config_path: Optional[Union[Path, str]] = None,
        env_file_path: Optional[Union[Path, str]] = None,
    ):
        self.config_path = (
            Path(config_path).expanduser()
            if config_path
            else Path.home() / ".config" / "valstorm" / "keys.json"
        )
        self.env_file_path = (
            Path(env_file_path).expanduser()
            if env_file_path
            else self._discover_repo_env_keys_path()
        )

    @staticmethod
    def _discover_repo_env_keys_path() -> Path:
        """Looks for .env.ai.keys or .env in current directory or parent git repo roots."""
        curr = Path.cwd().resolve()
        for directory in [curr, *curr.parents]:
            for filename in [".env.ai.keys", ".env"]:
                candidate = directory / filename
                if candidate.is_file():
                    return candidate
        return curr / ".env.ai.keys"

    @classmethod
    def _get_candidate_keys_for_provider(cls, provider: str) -> List[str]:
        """Returns ordered list of possible key names / env vars for a given provider."""
        prov_lower = provider.strip().lower()
        prov_upper = provider.strip().upper()

        mapped = cls.PROVIDER_ENV_MAP.get(prov_lower, [])
        keys = list(mapped)
        if f"{prov_upper}_API_KEY" not in keys:
            keys.append(f"{prov_upper}_API_KEY")
        if f"{prov_upper}_KEY" not in keys:
            keys.append(f"{prov_upper}_KEY")
        if prov_upper not in keys:
            keys.append(prov_upper)
        if prov_lower not in keys:
            keys.append(prov_lower)
        return keys

    def _read_json_config(self) -> Dict[str, str]:
        """Reads and parses the JSON config file if it exists."""
        if not self.config_path.is_file():
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items() if v}
        except Exception:
            return {}
        return {}

    def _read_env_keys_file(self) -> Dict[str, str]:
        """Reads and parses .env.ai.keys, .env, and local env files in workspace."""
        result: Dict[str, str] = {}
        curr = Path.cwd().resolve()
        candidate_paths = []
        for directory in [curr, *curr.parents]:
            for fname in [
                ".env.ai.keys",
                ".env",
                "apps/agent-runtime/.env",
                "apps/agent-runtime/.env.ai.keys",
                "apps/api/.env",
                "api-scripting/.env",
            ]:
                p = directory / fname
                if p.is_file():
                    candidate_paths.append(p)

        if self.env_file_path and Path(self.env_file_path).is_file():
            candidate_paths.append(Path(self.env_file_path))

        # Read in forward order and let the explicit env_file_path overwrite
        for path in candidate_paths:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        if "=" in line:
                            key, val = line.split("=", 1)
                            key = key.strip()
                            val = val.strip()
                            if val.startswith(('"', "'")):
                                quote_char = val[0]
                                closing_idx = val.find(quote_char, 1)
                                if closing_idx != -1:
                                    val = val[1:closing_idx]
                                else:
                                    val = val.strip("'\"")
                            else:
                                if " #" in val:
                                    val = val.split(" #", 1)[0].strip()
                                elif "\t#" in val:
                                    val = val.split("\t#", 1)[0].strip()
                                val = val.strip().strip("'\"")
                            if key and val:
                                result[key] = val
            except Exception:
                pass
        return result

    def get_api_key(
        self,
        provider: str,
        override_key: Optional[str] = None,
    ) -> Optional[str]:
        """Resolves the API key for the specified provider using the precedence hierarchy."""
        # 1. Override key
        if override_key and override_key.strip():
            return override_key.strip()

        candidate_keys = self._get_candidate_keys_for_provider(provider)

        # 2. Environment variables
        for candidate in candidate_keys:
            val = os.environ.get(candidate)
            if val and val.strip():
                return val.strip()

        # 3. Config JSON file (~/.config/valstorm/keys.json)
        json_keys = self._read_json_config()
        for candidate in candidate_keys:
            if candidate in json_keys and json_keys[candidate].strip():
                return json_keys[candidate].strip()
            if candidate.lower() in json_keys and json_keys[candidate.lower()].strip():
                return json_keys[candidate.lower()].strip()

        # 4. Repo .env.ai.keys file
        env_keys = self._read_env_keys_file()
        for candidate in candidate_keys:
            if candidate in env_keys and env_keys[candidate].strip():
                return env_keys[candidate].strip()
            if candidate.lower() in env_keys and env_keys[candidate.lower()].strip():
                return env_keys[candidate.lower()].strip()

        # 5. Defaults for local-only providers (ollama/vllm)
        if provider.lower() in ("ollama", "vllm"):
            return provider.lower()

        return None

    def save_api_key(
        self,
        provider: str,
        api_key: str,
        persist_to: str = "config",
    ) -> Path:
        """Saves an API key to ~/.config/valstorm/keys.json or .env.ai.keys."""
        clean_key = api_key.strip()
        prov_name = provider.strip().lower()

        if persist_to == "config":
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            existing = self._read_json_config()
            existing[prov_name] = clean_key
            with open(self.config_path, "w", encoding="utf-8") as f:
                json.dump(existing, f, indent=2)
            return self.config_path
        else:
            env_var = self.PROVIDER_ENV_MAP.get(prov_name, [f"{prov_name.upper()}_API_KEY"])[0]
            existing_lines = []
            replaced = False
            if self.env_file_path.is_file():
                with open(self.env_file_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip().startswith(f"{env_var}="):
                            existing_lines.append(f"{env_var}={clean_key}\n")
                            replaced = True
                        else:
                            existing_lines.append(line)
            if not replaced:
                existing_lines.append(f"{env_var}={clean_key}\n")
            with open(self.env_file_path, "w", encoding="utf-8") as f:
                f.writelines(existing_lines)
            return self.env_file_path

    @classmethod
    def resolve_key(
        cls,
        provider: str,
        override_key: Optional[str] = None,
        config_path: Optional[Union[Path, str]] = None,
        env_file_path: Optional[Union[Path, str]] = None,
    ) -> Optional[str]:
        """Convenience class method to resolve a key using a fresh KeyStore instance."""
        return cls(config_path=config_path, env_file_path=env_file_path).get_api_key(
            provider=provider,
            override_key=override_key,
        )
