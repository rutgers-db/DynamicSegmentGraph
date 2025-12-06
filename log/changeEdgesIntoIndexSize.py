def main():
    while True:
        try:
            n = float(input("Enter a number: "))
        except ValueError:
            print("Invalid input. Please enter a valid number.")
            continue

        # Size calculation: number * 10000 * 5 * 4 bytes
        total_bytes = n * 10000 * 5 * 4

        # Convert bytes to megabytes
        total_mb = total_bytes / (1024 * 1024)

        # Output the result
        print(f"The total size is {total_mb:.2f} MB")

if __name__ == "__main__":
    main()
