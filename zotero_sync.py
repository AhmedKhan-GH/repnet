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
items = zot.everything(zot.top())

# Filter to only show actual papers/books (not attachments/notes)
papers = [item for item in items if item['data'].get('itemType') not in ['attachment', 'note']]

print(f"\nFound {len(papers)} papers in group library (out of {len(items)} total items)")

import sys
from bibtexparser.bwriter import BibTexWriter

writer = BibTexWriter()

# If author name passed as argument, search and export BibTeX for that author
if len(sys.argv) > 1:
    search_name = sys.argv[1].lower()
    print(f"\nSearching for author: {search_name}")
    for item in papers:
        data = item['data']
        creators = data.get('creators', [])
        authors_lower = [c.get('lastName', '').lower() for c in creators if 'lastName' in c]
        if any(search_name in a for a in authors_lower):
            title = data.get('title', 'No title')
            print(f"\nFound: {title}")
            db = zot.item(item['key'], format='bibtex')
            print(writer.write(db))
    sys.exit(0)

# Export all BibTeX to project_references.bib (per-item to avoid commented entries)
print("\nGenerating BibTeX (per-item export)...")
all_bibtex = []
for item in papers:
    db = zot.item(item['key'], format='bibtex')
    bib_str = writer.write(db).strip()
    if bib_str and not bib_str.startswith('@comment'):
        all_bibtex.append(bib_str)

with open('project_references.bib', 'w', encoding='utf-8') as f:
    f.write('\n\n'.join(all_bibtex) + '\n')

print(f"BibTeX saved to project_references.bib ({len(all_bibtex)} entries)")

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
