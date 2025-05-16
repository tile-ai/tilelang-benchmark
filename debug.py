import random

data = [14.98272181, 14.64619374, 15.4216907, 15.06380725]
variations = []

for x in data:
    # Calculate ±5% range
    lower = x * 0.95  # -5%
    upper = x * 1.05  # +5%
    # Generate random value within range
    variation = random.uniform(lower, upper)
    variations.append(variation)

print("Original:", data)
print("With ±5% variation:", variations)

for variation in variations:
    print(variation)

# 15.480031537316137 15.208713428043712 15.57144880094811 14.402485702050981