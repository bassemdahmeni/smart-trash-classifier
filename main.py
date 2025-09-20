# configbox_demo.py
from box import ConfigBox

# Create a ConfigBox from a dictionary
user_data = {
    "name": "John Doe",
    "age": 30,
    "email": "john.doe@example.com",
    "address": {
        "street": "123 Main St",
        "city": "New York",
        "zipcode": "10001"
    },
    "hobbies": ["reading", "coding", "gaming"],
    "is_active": True
}

# Convert to ConfigBox
user = ConfigBox(user_data)

print("=== ConfigBox Demo ===")
print("Dot notation access:")
print(f"Name: {user.name}")
print(f"Age: {user.age}")
print(f"Email: {user.email}")
print(f"Address: {user.address.street}, {user.address.city} {user.address.zipcode}")
print(f"Hobbies: {', '.join(user.hobbies)}")
print(f"Active: {user.is_active}")

print("\n=== Nested access ===")
print(f"City: {user.address.city}")
print(f"Zipcode: {user.address.zipcode}")

print("\n=== Dictionary-style access still works ===")
print(f"Name (dict): {user['name']}")
print(f"City (dict): {user['address']['city']}")

print("\n=== Adding new attributes ===")
user.country = "USA"
user.address.state = "NY"
print(f"Country: {user.country}")
print(f"State: {user.address.state}")

print("\n=== Full ConfigBox content ===")
print(user)

print("\n=== Testing error handling ===")
try:
    print(user.nonexistent_field)
except AttributeError as e:
    print(f"Error (expected): {e}")