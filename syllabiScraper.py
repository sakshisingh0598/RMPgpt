import requests
from bs4 import BeautifulSoup
import os

branches = 'dsci'
# URL of the page to scrape
url = 'https://classes.usc.edu/term-20241/classes/' + branches

# Function to create a directory to save PDFs
def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

# Function to download PDF
def download_pdf(pdf_url, dest_folder):
    response = requests.get(pdf_url)
    if response.status_code == 200:
        file_name = pdf_url.split('/')[-1]
        file_path = os.path.join(dest_folder, file_name)
        with open(file_path, 'wb') as file:
            file.write(response.content)
        print(f'Downloaded: {file_name}')
    else:
        print(f'Failed to download: {pdf_url}')

# Request the page
response = requests.get(url)
if response.status_code == 200:
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all links that contain 'syllabus' in the href attribute
    syllabus_links = soup.find_all('a', href=lambda href: href and 'syllabus' in href)

    # Directory to save PDFs
    dest_folder = 'syllabus_pdfs_' + branches
    create_directory(dest_folder)

    # Loop through all syllabus links and download the PDFs
    for link in syllabus_links:
        pdf_url = link['href']
        if not pdf_url.startswith('http'):
            pdf_url = 'https://classes.usc.edu' + pdf_url
        download_pdf(pdf_url, dest_folder)
else:
    print(f'Failed to retrieve the page: {url}')
