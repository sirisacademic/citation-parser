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
        journal = data.get('primary_location', {}).get('source', {}).get('display_name', None)
        volume = data.get('biblio', {}).get('volume', None)
        issue = data.get('biblio', {}).get('issue', None)
        pages = f"{data.get('biblio', {}).get('first_page', '')}-{data.get('biblio', {}).get('last_page', '')}".strip("-")
        doi = data.get('doi', None)
        if doi:
            doi = doi.replace("https://doi.org/", "")

        citation_parts = [
            f"{authors} ({year}).",
            f"{title}.",
            f"{journal}," if journal else "",
            f"{volume}" if volume or issue else "",
            f"{pages}." if pages else "",
            f"doi: {doi}" if doi else ""
        ]
        return " ".join(part for part in citation_parts if part)

class OpenAIREFormatter(CitationFormatter):
    def generate_apa_citation(self, data):
        """
        Generate an APA-style citation from OpenAIRE JSON data.

        :param data: OpenAIRE JSON data
        :return: APA citation string
        """
        # Extract authors
        creators = data.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('creator', [])
        authors_list = [
            f"{creator['$'].split(' ')[1]}, {creator['$'].split(' ')[0][0]}."
            for creator in creators
        ]
        if len(authors_list) > 3:
            authors = ", ".join(authors_list[:3]) + ", et al."
        else:
            authors = ", ".join(authors_list)

        # Extract title
        title = data.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {}).get('title', {})[0].get('$', "Unknown Title")

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
        identifiers = data.get('metadata', {}).get('oaf:entity', {}).get('oaf:result', {})['pid']
        doi = None
        for identifier in identifiers:
            if identifier.get('@classid') == 'doi':
                doi =identifier.get('$',None)
                doi = doi.replace("https://doi.org/", "")


        # Construct the APA citation
        citation_parts = [
            f"{authors} ({year}).",
            f"{title}.",
            f"{journal}," if journal else "",
            f"{volume}" if volume or issue else "",
            f"{pages}." if pages else "",
            f"doi: {doi}" if doi else ""
        ]
        # Join non-empty parts with spaces
        citation = " ".join(part for part in citation_parts if part)

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

class CitationFormatterFactory:
    formatters = {
        "openalex": OpenAlexFormatter(),
        "openaire": OpenAIREFormatter(),
        "pubmed": PubMedFormatter(),
    }

    @staticmethod
    def get_formatter(api_name):
        if api_name not in CitationFormatterFactory.formatters:
            raise ValueError(f"Unsupported API: {api_name}")
        return CitationFormatterFactory.formatters[api_name]