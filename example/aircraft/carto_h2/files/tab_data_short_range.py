"""Extract all the data listed in data_to_capture from JSON files and returns a Latex table"""

from marilib.utils.read_write import MarilibIO

filename = "C:\\Users\\monrolni\\Documents\\Python\\MARILib_obj\\example\\best_airplane_ever.json"

io = MarilibIO()
aircraft_data = io.from_json_file(filename)



print(aircraft_data)
