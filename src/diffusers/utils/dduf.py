import zipfile


class DDUFReader:
    def __init__(self, dduf_file):
        self.dduf_file = dduf_file
        self.files = []
        self.post_init()

    def post_init(self):
        """
        Check that the DDUF file is valid
        """
        if not zipfile.is_zipfile(self.dduf_file):
            raise ValueError(f"The file '{self.dduf_file}' is not a valid ZIP archive.")

        try:
            with zipfile.ZipFile(self.dduf_file, "r") as zf:
                # Check integrity and store file list
                zf.testzip()  # Returns None if no corrupt files are found
                self.files = zf.namelist()
        except zipfile.BadZipFile:
            raise ValueError(f"The file '{self.dduf_file}' is not a valid ZIP archive.")
        except Exception as e:
            raise RuntimeError(f"An error occurred while validating the ZIP file: {e}")

    def has_file(self, file):
        return file in self.files

    def read_file(self, file_name, encoding=None):
        """
        Reads the content of a specific file in the ZIP archive without extracting.
        """
        if file_name not in self.files:
            raise ValueError(f"{file_name} is not in the list of files {self.files}")
        with zipfile.ZipFile(self.dduf_file, "r") as zf:
            with zf.open(file_name) as file:
                file = file.read()
                if encoding is not None:
                    file = file.decode(encoding)
                return file
