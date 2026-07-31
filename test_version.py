import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from genai_proxy.version import (
    LOCAL_DEV_VERSION,
    ProgramVersion,
    format_program_version,
    get_program_version,
    write_build_metadata,
)


class ProgramVersionTests(unittest.TestCase):
    def test_reads_baked_image_metadata_before_git(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "version.json"
            metadata_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "commit": "a" * 40,
                        "committed_at": "2026-07-31T12:34:56+08:00",
                    }
                ),
                encoding="utf-8",
            )

            version = get_program_version(
                metadata_path=metadata_path,
                source_root=Path(temp_dir) / "not-a-repository",
            )

        self.assertEqual(
            version,
            ProgramVersion(
                "a" * 40,
                "2026-07-31T12:34:56+08:00",
                "image",
            ),
        )

    def test_reads_full_hash_and_commit_time_from_current_git_checkout(self):
        repository_root = Path(__file__).resolve().parent
        missing_metadata = repository_root / "missing-build-version.json"
        version = get_program_version(
            metadata_path=missing_metadata,
            source_root=repository_root,
        )
        expected = subprocess.run(
            ["git", "log", "-1", "--format=%H%n%cI"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()

        self.assertEqual(version.commit, expected[0])
        self.assertEqual(version.committed_at, expected[1])
        self.assertEqual(version.source, "git")

    def test_invalid_metadata_and_missing_git_fall_back_to_local_dev(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "version.json"
            metadata_path.write_text('{"version":1,"commit":"short"}', encoding="utf-8")

            version = get_program_version(
                metadata_path=metadata_path,
                source_root=temp_dir,
            )

        self.assertEqual(version.commit, LOCAL_DEV_VERSION)
        self.assertIsNone(version.committed_at)
        self.assertEqual(format_program_version(version), LOCAL_DEV_VERSION)

    def test_build_arguments_are_written_for_gitless_image_builds(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "version.json"
            version = write_build_metadata(
                output_path,
                source_root=temp_dir,
                environ={
                    "GENAI_BUILD_COMMIT": "B" * 40,
                    "GENAI_BUILD_COMMIT_TIME": "2026-07-31T03:04:05Z",
                    "GENAI_BUILD_DIRTY": "true",
                },
            )
            payload = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(version.commit, "b" * 40)
        self.assertEqual(version.source, "build-args")
        self.assertTrue(version.dirty)
        self.assertEqual(
            payload,
            {
                "version": 1,
                "commit": "b" * 40,
                "committed_at": "2026-07-31T03:04:05Z",
                "dirty": True,
            },
        )

    def test_invalid_dirty_build_argument_falls_back_to_local_dev(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "version.json"
            version = write_build_metadata(
                output_path,
                source_root=temp_dir,
                environ={
                    "GENAI_BUILD_COMMIT": "b" * 40,
                    "GENAI_BUILD_COMMIT_TIME": "2026-07-31T03:04:05Z",
                    "GENAI_BUILD_DIRTY": "invalid",
                },
            )

        self.assertTrue(version.is_local_dev)

    def test_local_dev_metadata_is_valid_image_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "version.json"
            written = write_build_metadata(
                output_path,
                source_root=temp_dir,
                environ={},
            )
            loaded = get_program_version(
                metadata_path=output_path,
                source_root=Path(temp_dir) / "missing",
            )

        self.assertTrue(written.is_local_dev)
        self.assertEqual(loaded, ProgramVersion(LOCAL_DEV_VERSION, None, "image"))

    def test_dirty_source_is_explicit_in_formatted_version(self):
        version = ProgramVersion(
            "c" * 40,
            "2026-07-31T03:04:05+00:00",
            "git",
            dirty=True,
        )

        self.assertEqual(
            format_program_version(version),
            (
                f"commit={'c' * 40} "
                "committed_at=2026-07-31T03:04:05+00:00 "
                "source=git dirty=true"
            ),
        )

    def test_deployment_configuration_changes_mark_checkout_dirty(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = Path(temp_dir)
            (repository / "src" / "genai_proxy").mkdir(parents=True)
            (repository / "src" / "genai_proxy" / "__init__.py").write_text(
                "\n",
                encoding="utf-8",
            )
            (repository / "docker-compose.yml").write_text(
                "services: {}\n",
                encoding="utf-8",
            )
            for command in (
                ("git", "init", "-q"),
                ("git", "config", "user.name", "Version Test"),
                ("git", "config", "user.email", "version@example.test"),
                ("git", "add", "."),
                ("git", "-c", "commit.gpgsign=false", "commit", "-qm", "initial"),
            ):
                subprocess.run(command, cwd=repository, check=True)

            clean = get_program_version(
                metadata_path=repository / "missing.json",
                source_root=repository,
            )
            (repository / "docker-compose.yml").write_text(
                "services:\n  proxy: {}\n",
                encoding="utf-8",
            )
            dirty = get_program_version(
                metadata_path=repository / "missing.json",
                source_root=repository,
            )

        self.assertFalse(clean.dirty)
        self.assertTrue(dirty.dirty)


if __name__ == "__main__":
    unittest.main()
