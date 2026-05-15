import os
import zipfile
import tarfile
from pathlib import Path


class FilesGetter:

    @staticmethod
    def is_valid_path(path: str) -> bool:
        for value in path.split(sep='/'):
            if value.startswith(('.', '__')):
                return False
        return True

    def extract_files_data(self, source_path: str) -> list[tuple[str, str]]:
        files_data = []
        path_obj = Path(source_path)

        if path_obj.is_dir():
            for root, _, files in os.walk(source_path):
                if not self.is_valid_path(root):
                    continue
                for file in files:
                    full_path = os.path.join(root, file)
                    try:
                        with open(full_path, 'r', encoding='utf-8') as f:
                            files_data.append((full_path, f.read()))
                    except UnicodeDecodeError:
                        pass
                    except Exception as err:
                        print(f"Can't open {full_path}: {err}")

        elif source_path.endswith('.zip'):
            with zipfile.ZipFile(source_path, 'r') as z:
                for file_info in z.infolist():
                    if (not file_info.is_dir()
                            and self.is_valid_path(file_info.name)):
                        try:
                            content = z.read(
                                file_info.filename).decode('utf-8')
                            files_data.append((file_info.filename, content))
                        except UnicodeDecodeError:
                            pass
                        except Exception as err:
                            print(
                                f"Error on {file_info.filename}: {err}")

        elif source_path.endswith(('.tar', '.tar.gz', '.tgz', '.tar.bz2')):
            with tarfile.open(source_path, 'r:*') as t:
                for member in t.getmembers():
                    if member.isfile() and self.is_valid_path(member.name):
                        try:
                            f = t.extractfile(member)
                            if f is not None:
                                content = f.read().decode('utf-8')
                                files_data.append((member.name, content))
                        except UnicodeDecodeError:
                            pass  # Fichier binaire, on ignore
                        except Exception as e:
                            print(f"Error on {member.name} : {e}")

        else:
            print(f"Format not supported for {source_path}.")
            print("Please input a file, a .zip or a .tar.")

        return files_data
