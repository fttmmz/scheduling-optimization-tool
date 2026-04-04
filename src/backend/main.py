from database.loader import DataLoader
if __name__ == "__main__":
    loader = DataLoader()
    data = loader.load_all()

    print(data["courses"][0].name)