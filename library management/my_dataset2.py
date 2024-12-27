import pandas as pd
import random
books = pd.DataFrame({
    'Book_ID': range(1, 101),
    'Genre': random.choices(['Fiction', 'Non-Fiction', 'Sci-Fi', 'Romance'], k=100),
    'Author': random.choices(['Author_A', 'Author_B', 'Author_C'], k=100),
    'Publication_Year': random.choices(range(1980, 2023), k=100)
})
borrowers = pd.DataFrame({
    'Borrower_ID': range(1, 51),
    'Membership_Type': random.choices(['Regular', 'Premium'], k=50),
    'Age': random.choices(range(18, 70), k=50)
})
records = pd.DataFrame({
    'Record_ID': range(1, 201),
    'Book_ID': random.choices(books['Book_ID'], k=200),
    'Borrower_ID': random.choices(borrowers['Borrower_ID'], k=200),
    'Date_Borrowed': pd.date_range(start='2023-01-01', periods=200, freq='D')
})
books.to_csv('books.csv', index=False)
borrowers.to_csv('borrowers.csv', index=False)
records.to_csv('records.csv', index=False)

print("Datasets have been saved as 'books.csv', 'borrowers.csv', and 'records.csv'.")
