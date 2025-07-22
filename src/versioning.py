"""
Automated experiment versioning system.
Handles auto-incrementing version numbers and directory management.
"""

import os
import re
import json
from typing import Dict, Any, Optional
from datetime import datetime

class ExperimentVersionManager:
    """Manages experiment versions and directory structure."""
    
    def __init__(self, results_base_dir: str = "../results"):
        self.results_base_dir = results_base_dir
        os.makedirs(results_base_dir, exist_ok=True)
    
    def get_next_version(self) -> str:
        """Get the next available version number (e.g., 'v1', 'v2', 'v3')."""
        existing_versions = self._get_existing_versions()
        if not existing_versions:
            return "v1"
        
        max_version = max(existing_versions)
        next_num = int(max_version[1:]) + 1
        return f"v{next_num}"
    
    def _get_existing_versions(self) -> list[str]:
        """Get list of existing version directories."""
        if not os.path.exists(self.results_base_dir):
            return []
        
        versions = []
        for item in os.listdir(self.results_base_dir):
            item_path = os.path.join(self.results_base_dir, item)
            if os.path.isdir(item_path) and re.match(r'^v\d+$', item):
                versions.append(item)
        
        return sorted(versions, key=lambda x: int(x[1:]))
    
    def create_version_dir(self, version: str) -> str:
        """Create and return the path for a version directory."""
        version_dir = os.path.join(self.results_base_dir, version)
        os.makedirs(version_dir, exist_ok=True)
        return version_dir
    
    def get_versioned_path(self, filename: str, version: str) -> str:
        """Get a versioned file path in the results directory."""
        version_dir = self.create_version_dir(version)
        return os.path.join(version_dir, filename)
    
    def log_experiment_config(self, version: str, config: Dict[str, Any]) -> str:
        """Log experiment configuration to version directory."""
        config_with_metadata = {
            **config,
            'version': version,
            'timestamp': datetime.now().isoformat(),
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        config_path = self.get_versioned_path("config.json", version)
        with open(config_path, 'w') as f:
            json.dump(config_with_metadata, f, indent=2)
        
        return config_path

# Global instance
_version_manager = None

def get_version_manager() -> ExperimentVersionManager:
    """Get the global version manager instance."""
    global _version_manager
    if _version_manager is None:
        _version_manager = ExperimentVersionManager()
    return _version_manager

def get_next_version() -> str:
    """Get the next available experiment version."""
    return get_version_manager().get_next_version()

def get_versioned_path(filename: str, version: str = None) -> str:
    """Get a versioned file path. If version is None, uses next available version."""
    if version is None:
        version = get_next_version()
    return get_version_manager().get_versioned_path(filename, version)

def create_experiment_session() -> tuple[str, str]:
    """Start a new experiment session and return (version, version_dir)."""
    vm = get_version_manager()
    version = vm.get_next_version()
    version_dir = vm.create_version_dir(version)
    return version, version_dir

def log_experiment_config(config: Dict[str, Any], version: str = None) -> str:
    """Log experiment configuration. If version is None, uses next available version."""
    if version is None:
        version = get_next_version()
    return get_version_manager().log_experiment_config(version, config)