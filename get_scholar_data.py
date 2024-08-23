from scholarly import scholarly


def get_author_result(author_name):
    search_query = scholarly.search_author(author_name)
    # Retrieve the first result from the iterator
    first_author_result = next(search_query)
    interests = first_author_result['interests']
    author = scholarly.fill(first_author_result)
    publications = author['publications']
    clean_publications = []
    for index, publication in enumerate(publications):
        title = publication['bib']['title']
        try:
            year = int(publication['bib']['pub_year'])
        except KeyError:
            year = 0
        try:
            citations = publication['num_citations']
        except KeyError:
            citations = 0
        clean_publications.append([index, title, year, citations])

    return interests, publications, clean_publications


def sort_by_year(publication_list, newest_first=True):
    return sorted(publication_list, key=lambda x: x[2], reverse=newest_first)


def sort_by_citations(publication_list, highest_first=True):
    return sorted(publication_list, key=lambda x: x[3], reverse=highest_first)


def get_highest_n_lowest(function, amount):
    a = function
    if len(a) < 2*amount:
        return a
    else:
        result = a[:amount]+a[len(a)-amount:]
        return result
