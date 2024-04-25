import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

class FirebaseUtility:
    def __init__(self, credential_path, database_url):
        cred = credentials.Certificate(credential_path)
        firebase_admin.initialize_app(cred, {'databaseURL': database_url})
        self.db = db.reference()

    def add_table(self, table_name):
        try:
            self.db.child(table_name).set({})
            print(f"table '{table_name}' added successfully.")
        except Exception as e:
            print("Error adding table:", e)

    def put_data(self, table_name, data):
        try:
            self.db.child(table_name).push(data)
            print("data added successfully.")
        except Exception as e:
            print("Error putting data:", e)

    def get_data(self, table_name, attribute):
        try:
            data = self.db.child(table_name).get()
            if data is None:
                return None
            for key, value in data.items():
                if attribute in value:
                    return value[attribute]
            return None
        except Exception as e:
            print("Error getting data:", e)
            return None

    def get_all_data(self, table_name):
        try:
            return self.db.child(table_name).get()
        except Exception as e:
            print("Error getting all data:", e)
            return None


