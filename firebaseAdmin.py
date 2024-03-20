import firebase_admin
from firebase_admin import credentials, db

class FirebaseAdmin:
    def __init__(self, service_account_key_path, db_ref):
        cred = credentials.Certificate(service_account_key_path)
        firebase_admin.initialize_app(cred)
        self.db_ref = db_ref  # Reference to the Realtime Database

    def read_data(self, path):
        """Read data from the Realtime Database."""
        return self.db_ref.child(path).get()

    def write_data(self, path, data):
        """Write data to the Realtime Database."""
        self.db_ref.child(path).set(data)

    # You can define more functions for updating, deleting, querying data, etc.

