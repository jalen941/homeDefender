import firebase_admin
from firebase_admin import credentials, db

class FirebaseAdmin:
    def __init__(self, service_account_key_path, db_ref):
        cred = credentials.Certificate(service_account_key_path)
        firebase_admin.initialize_app(cred)
        self.db_ref = db_ref

    def read_data(self, path):
        """read data from the Realtime Database."""
        return self.db_ref.child(path).get()

    def write_data(self, path, data):
        """write data to the Realtime Database."""
        self.db_ref.child(path).set(data)



