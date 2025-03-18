# citation_formatter.py
import xml.etree.ElementTree as ET

# CitationFormatter Base Class
class CitationFormatter:
    def generate_apa_citation(self, data):
        raise NotImplementedError("Subclasses must implement this method.")
    
class OpenAlexFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        
        authors_list = [auth['raw_author_name'] for auth in data.get('authorships', [])]
        if len(authors_list) > 3:
            authors = ", ".join(authors_list[:3]) + ", et al."
        else:
            authors = ", ".join(authors_list)

        title = data.get('title', "Unknown Title")
        year = data.get('publication_year', "n.d.")

        try:
            journal = data.get('primary_location', {}).get('source', {}).get('display_name', None)
        except AttributeError:
            journal = None

        try:
            volume = data.get('biblio', {}).get('volume', None)
        except AttributeError:
            volume = None

        try:
            issue = data.get('biblio', {}).get('issue', None)
        except AttributeError:
            issue = None

        try:
            first_page = data.get('biblio', {}).get('first_page', '')
            last_page = data.get('biblio', {}).get('last_page', '')
            pages = f"{first_page}-{last_page}".strip("-")
        except AttributeError:
            pages = ""

        try:
            doi = data.get('doi', None)
            if doi:
                doi = doi.replace("https://doi.org/", "")
        except AttributeError:
            doi = None

        citation_parts = [
            f"{authors}" if authors else "",
            f"({year})." if authors else ".",
            f"{title}." if title else "",
            f"{journal}," if journal else "",
            f"{volume}" if volume else "",
            f"({issue})" if issue else "",
            f"{pages}." if pages else "",
            f"doi: {doi}" if doi else ""
        ]

        return " ".join(part for part in citation_parts if part).strip()


class OpenAIREFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        """
        Generate an APA-style citation from OpenAIRE JSON data.

        :param data: OpenAIRE JSON data
        :return: APA citation string
        """
        # Extract authors
        creators = data.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('creator', [])

        authors_list = []
        for creator in creators:
            # Ensure `creator` is a dictionary and contains '$'
            if not isinstance(creator, dict) or '$' not in creator:
                print(f"Skipping invalid creator entry: {creator}")  # Debugging line
                continue  # Skip invalid data

            name_parts = creator['$'].split()

            if len(name_parts) == 1:
                formatted_name = name_parts[0]  # Single-word name
            elif len(name_parts) == 2:
                formatted_name = f"{name_parts[1]}, {name_parts[0][0]}."
            else:
                formatted_name = f"{name_parts[-1]}, {''.join([p[0] for p in name_parts[:-1]])}."

            authors_list.append(formatted_name)

        # Formatting the final authors string
        if len(authors_list) > 3:
            authors = ", ".join(authors_list[:3]) + ", et al."
        else:
            authors = ", ".join(authors_list)


        # Extract title
        title_data = data.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('title', {})

        if isinstance(title_data, list) and title_data:  
            # If 'title' is a list and not empty, get the first element's '$' value
            title = title_data[0].get('$', "Unknown Title")
        elif isinstance(title_data, dict):  
            # If 'title' is a dictionary, directly get its '$' value
            title = title_data.get('$', "Unknown Title")
        else:
            title = None

        # Extract publication year
        date_of_acceptance = data.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('dateofacceptance', {}).get('$', None)
        year = date_of_acceptance.split("-")[0] if date_of_acceptance else "n.d."

        # Extract journal name (not explicitly available in OpenAIRE data, so use placeholder)
        journal = data.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('journal', {}).get('$', None)

        # Extract volume, issue, and pages (if available)
        volume = None  # Placeholder since OpenAIRE schema does not directly include volume
        issue = None   # Placeholder since OpenAIRE schema does not directly include issue
        pages = None   # Placeholder since OpenAIRE schema does not directly include page range

        # Extract DOI
        identifiers = data.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('pid', [])
        # Ensure identifiers is always a list
        if isinstance(identifiers, dict):  
            identifiers = [identifiers]  
        doi = None
        for identifier in identifiers:
            if identifier.get('@classid') == 'doi':
                doi =identifier.get('$',None)
                doi = doi.replace("https://doi.org/", "")


        # Construct the APA citation, including only non-empty parts
        citation_parts = [
            f"{authors}" if authors else "",
            f"({year})." if year else ".",
            f"{title}." if title else "",
            f"{journal}," if journal else "",
            f"{volume}" if volume else "",
            f"({issue})" if issue else "",
            f"{pages}." if pages else "",
            f"doi: {doi}" if doi else ""
        ]
        # Join non-empty parts with spaces
        citation = " ".join(part for part in citation_parts if part).strip()

        return citation
    
class PubMedFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        # Parse the XML data
        root = ET.fromstring(data)
        article = root.find("PubmedArticle/MedlineCitation/Article")

        # Extract authors
        authors = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last_name = author.findtext("LastName", "")
                fore_name = author.findtext("ForeName", "")
                initials = author.findtext("Initials", "")
                authors.append(f"{last_name}, {initials or fore_name[0] if fore_name else ''}.")

        if len(authors) > 3:
            authors = ", ".join(authors[:3]) + ", et al."
        else:
            authors = ", ".join(authors)

        # Extract title
        title = article.findtext("ArticleTitle", "Unknown Title")

        # Extract year
        pub_date = article.find("Journal/JournalIssue/PubDate")
        year = pub_date.findtext("Year", "n.d.") if pub_date is not None else "n.d."

        # Extract journal
        journal = article.findtext("Journal/Title", None)

        # Extract volume, issue, and pages
        volume = article.findtext("Journal/JournalIssue/Volume", None)
        issue = article.findtext("Journal/JournalIssue/Issue", None)
        start_page = article.findtext("Pagination/StartPage", "")
        end_page = article.findtext("Pagination/EndPage", "")
        pages = f"{start_page}-{end_page}".strip("-")

        # Extract DOI
        doi = article.findtext("ELocationID[@EIdType='doi']", None)

        citation_parts = [
            f"{authors} ({year}).",
            f"{title}.",
            f"{journal}," if journal else "",
            f"{volume}({issue})" if volume or issue else "",
            f"{pages}." if pages else "",
            f"doi: {doi}" if doi else ""
        ]
        return " ".join(part for part in citation_parts if part)

class CrossrefFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        # Extracting authors
        authors_list = [f"{auth.get('given', '')} {auth.get('family', '')}".strip() for auth in data.get('author', [])]

        if len(authors_list) > 3:
            authors = ", ".join(authors_list[:3]) + ", et al."
        else:
            authors = ", ".join(authors_list)

        # Title of the paper
        title = data.get('title', ["Unknown Title"])[0]

        # Publication year
        year = data.get('issued', {}).get('date-parts', [[None]])[0][0] or "n.d."

        # Publisher info
        publisher = data.get('publisher', "")
        publisher_location = data.get('container-title', "Unknown Location")[0] or ''

        # Pages range (if available)
        pages = data.get('page', "")
        pages_range = f"{pages}" if pages else ""

        # DOI (if available)
        doi = data.get('DOI', None)
        if doi:
            doi = doi.replace("https://doi.org/", "")

        # Format the citation in APA style
        citation_parts = [
            f"{authors} ({year}).",
            f"{title}.",
            f"{publisher_location}: {publisher}.",
            f"{pages_range}." if pages_range else "",
            f"doi: {doi}" if doi else ""
        ]
        return " ".join(part for part in citation_parts if part)

class HALFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        # Extract authors
        author_name = data.get('authFullName_s', "")
        authors_list = author_name
        if len(authors_list) > 3:
            authors = ", ".join(authors_list[:3]) + ", et al."
        else:
            authors = ", ".join(authors_list)

        # Extract title
        title = data.get('title_s', [""])[0]

        # Extract year
        year = data.get('conferenceStartDateY_i', data.get('publicationStartDateY_i', data.get('publicationDateY_i', "")))

        # Extract book/journal title
        book_title = data.get('journalTitle_s', data.get('bookTitle_s', None))

        # Extract pages
        pages = data.get('page_s', None)
        doi = data.get('doiId_s', None)
        volume = data.get('volume_s', None)
        issue = data.get('issue_s', [None])[0]

        # Construct citation
        citation_parts = [
            f"{authors}" if authors else "",
            f"({year})." if year else ".",
            f"{title}." if title else "",
            f"{book_title}," if book_title else "",
            f"{volume}" if volume else "",
            f"({issue})" if issue else "",
            f"pp. {pages}." if pages else "",
            f"doi: {doi}" if doi else ""
        ]

        # Return formatted citation
        return " ".join(part for part in citation_parts if part).strip()


class CitationFormatterFactory:
    formatters = {
        "openalex": OpenAlexFormatter(),
        "openaire": OpenAIREFormatter(),
        "pubmed": PubMedFormatter(),
        "crossref": CrossrefFormatter(),
        "hal": HALFormatter(),
    }

    @staticmethod
    def get_formatter(api_name):
        if api_name not in CitationFormatterFactory.formatters:
            raise ValueError(f"Unsupported API: {api_name}")
        return CitationFormatterFactory.formatters[api_name]