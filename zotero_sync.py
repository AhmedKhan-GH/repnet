import os
from dotenv import load_dotenv
from pyzotero import zotero

# Load environment variables
load_dotenv()

# Get credentials
group_id = os.getenv('ZOTERO_GROUP_ID')
api_key = os.getenv('ZOTERO_API_KEY')

print(f"Connecting to group: {group_id}")

# Try to connect to personal library
print("\nTrying personal library...")
try:
    personal_zot = zotero.Zotero(group_id, 'user', api_key)
    personal_items = personal_zot.top(limit=10)
    print(f"Personal library access: SUCCESS - found {len(personal_items)} items")
except Exception as e:
    print(f"Personal library access: FAILED - {e}")

# Connect to Zotero GROUP (not personal library)
zot = zotero.Zotero(group_id, 'group', api_key)

# Fetch items (excluding attachments and notes)
items = zot.top(limit=100)

# Filter to only show actual papers/books (not attachments/notes)
papers = [item for item in items if item['data'].get('itemType') not in ['attachment', 'note']]

print(f"\nFound {len(papers)} papers in group library (out of {len(items)} total items)")

# Get BibTeX for all items
print("\nGenerating BibTeX...")
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

bibtex_db = zot.items(format='bibtex')
writer = BibTexWriter()
bibtex_str = writer.write(bibtex_db)

# Save to file
with open('references.bib', 'w', encoding='utf-8') as f:
    f.write(bibtex_str)

print(f"✓ BibTeX saved to references.bib")

print("\nPapers:")
for i, item in enumerate(papers, 1):
    data = item['data']
    item_type = data.get('itemType', 'unknown')
    title = data.get('title', 'No title')
    creators = data.get('creators', [])
    authors = ', '.join([c.get('lastName', '') for c in creators if 'lastName' in c])
    year = data.get('date', 'No date')[:4] if data.get('date') else 'No year'

    print(f"{i}. [{item_type}] {title}")
    print(f"   Authors: {authors}")
    print(f"   Year: {year}")
    print()
