import sounddevice as sd


def list_microphones():
    return [
        (index, device["name"])
        for index, device in enumerate(sd.query_devices())
        if device["max_input_channels"] > 0
    ]


def select_microphone():
    
    microphones = list_microphones()
    if not microphones:
        raise RuntimeError("Nincs elerheto hangbemenet.")

    for number, (_, name) in enumerate(microphones, start=1):
        print(f"[{number}] {name}")

    while True:
        selection = input("Mikrofon sorszama [0 = vissza]: ").strip()
        if selection == "0":
            return None
        try:
            number = int(selection)
        except ValueError:
            number = 0
        if 1 <= number <= len(microphones):
            return microphones[number - 1][0]
        print("Ervenytelen sorszam.")
