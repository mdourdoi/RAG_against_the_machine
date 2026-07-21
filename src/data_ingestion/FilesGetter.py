import os
import zipfile
from pathlib import Path


class FilesGetter:
    """Collect supported source files from a directory or zip archive."""

    @staticmethod
    def is_valid_path(path: str) -> bool:
        """Return whether a path should be ingested.

        Args:
            path: File or directory path to validate.

        Returns:
            True when no path component starts with a dot.
        """
        for value in path.split(sep='/'):
            if value.startswith(('.')):
                return False
        return True

    def extract_files_data(self, source_path: str) -> list[tuple[str, str]]:
        """Read supported files and return their paths and contents.

        Args:
            source_path: Directory or zip archive to scan.

        Returns:
            A list of ``(file_path, content)`` tuples for supported files.
        """
        files_data = []
        path_obj = Path(source_path)

        if path_obj.is_dir():
            for root, _, files in os.walk(source_path):
                if not self.is_valid_path(root):
                    continue
                for file in files:
                    full_path = os.path.join(root, file).replace('\\', '/')
                    if not (
                        full_path.endswith(
                            ('.py', '.md', '.txt', '.yaml', '.sh'))):
                        continue
                    try:
                        with open(full_path, 'r', encoding='utf-8',
                                  newline='') as f:
                            files_data.append(
                                (full_path, f.read().replace('\r\n', '\n')))
                    except UnicodeDecodeError:
                        pass
                    except Exception as err:
                        print(f"Can't open {full_path}: {err}")

        elif source_path.endswith('.zip'):
            try:
                with zipfile.ZipFile(source_path, 'r') as z:
                    for file_info in z.infolist():
                        if (not file_info.is_dir()
                                and self.is_valid_path(file_info.filename)):
                            if not file_info.filename.endswith(
                                    ('.py', '.md', '.txt', '.yaml', '.sh')):
                                continue
                            try:
                                content = z.read(
                                    file_info.filename).decode('utf-8')
                                content = content.replace('\r\n', '\n')
                                res = str(path_obj.parent / file_info.filename)
                                files_data.append((res, content))
                            except UnicodeDecodeError:
                                pass
                            except Exception as err:
                                print(
                                    f"Error on {file_info.filename}: {err}")
            except Exception:
                raise Exception(f"Can't open {source_path}")

        else:
            print(f"Format not supported for {source_path}.")
            print("Please input a file or a .zip, or ensure the file exists.")

        return files_data
