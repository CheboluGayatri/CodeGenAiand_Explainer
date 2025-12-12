import asyncio
import csv
import matplotlib.pyplot as plt
from collections import defaultdict

DATA_CSV = "test_dataset.csv"

# Example model outputs (replace with actual API responses or local model calls)
MODEL_OUTPUTS = {
    "llama3_8b": {
        "Write a Python function to calculate factorial.": "def factorial(n): return 1 if n==0 else n*factorial(n-1)",
        "Write a Python function to reverse a string.": "def reverse_string(s): return s[::-1]",
        "Write a C program to swap two numbers.": "int a,b,temp; temp=a; a=b; b=temp;",
        "Write a C program to check if a number is even or odd.": "if(num%2==0){printf(\"even\");}else{printf(\"odd\");}",
        "Write a Java program to print \"Hello World\".": "System.out.println(\"Hello World\");",
    },
    "deepseek_6_7b": {
        "Write a Python function to calculate factorial.": "def factorial(n): return 1 if n==0 else n*factorial(n-1)",
        "Write a Python function to reverse a string.": "def reverse_string(s): return s[::-1]",
        "Write a C program to swap two numbers.": "temp=a; a=b; b=temp;",
        "Write a C program to check if a number is even or odd.": "if(num%2==0){printf(\"even\");}else{printf(\"odd\");}",
        "Write a Java program to print \"Hello World\".": "System.out.println(\"Hello World\");",
    },
    "gemma_2b": {
        "Write a Python function to calculate factorial.": "def factorial(n): return 1 if n==0 else n*factorial(n-1)",
        "Write a Python function to reverse a string.": "def reverse_string(s): return s[::-1]",
        "Write a C program to swap two numbers.": "int a,b,temp; temp=a; a=b; b=temp;",
        "Write a C program to check if a number is even or odd.": "if(num%2==0){printf(\"even\");}else{printf(\"odd\");}",
        "Write a Java program to print \"Hello World\".": "System.out.println(\"Hello World\");",
    },
    "gemini_api": {
        "Write a Python function to calculate factorial.": "def factorial(n): return 1 if n<=1 else n*factorial(n-1)",
        "Write a Python function to reverse a string.": "def reverse_string(s): return s[::-1]",
        "Write a C program to swap two numbers.": "int temp=a; a=b; b=temp;",
        "Write a C program to check if a number is even or odd.": "if(num%2==0){printf(\"even\");}else{printf(\"odd\");}",
        "Write a Java program to print \"Hello World\".": "System.out.println(\"Hello World\");",
    },
    "groq_api": {
        "Write a Python function to calculate factorial.": "def fact(n): return 1 if n==0 else n*fact(n-1)",
        "Write a Python function to reverse a string.": "def reverse_string(s): return s[::-1]",
        "Write a C program to swap two numbers.": "int temp=a; a=b; b=temp;",
        "Write a C program to check if a number is even or odd.": "if(num%2==0){printf(\"even\");}else{printf(\"odd\");}",
        "Write a Java program to print \"Hello World\".": "System.out.println(\"Hello World\");",
    },
}

async def evaluate_all():
    rows = []
    with open(DATA_CSV, newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    results = defaultdict(list)

    for row in rows:
        prompt = row["prompt"]
        expected = row["expected_output"].strip()
        for model, outputs in MODEL_OUTPUTS.items():
            predicted = outputs.get(prompt, "").strip()
            correct = 1 if predicted == expected else 0
            results[model].append(correct)

    return results

async def main():
    results = await evaluate_all()

    accuracies = {m: sum(v) / len(v) * 100 for m, v in results.items()}

    print("\nModel Accuracy Results:")
    for model, acc in accuracies.items():
        print(f"{model}: {acc:.2f}%")

    # Plot accuracy chart
    plt.figure(figsize=(8, 4))
    max_acc = max(accuracies.values())
    colors = ["green" if acc == max_acc else "lightseagreen" for acc in accuracies.values()]
    bars = plt.bar(accuracies.keys(), accuracies.values(), color=colors)
    plt.ylim(0, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Model Accuracy Comparison (Exact Match)")
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Add accuracy values on top
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, h + 1, f"{h:.1f}%", ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("model_accuracy_comparison.png", dpi=150)
    plt.show()
    print("\n✅ Saved chart: model_accuracy_comparison.png")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user. Exiting gracefully...")