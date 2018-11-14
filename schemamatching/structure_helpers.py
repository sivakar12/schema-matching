import re

def get_only_children_tags(tags):
    output_tags = set(tags)
    for tag in tags:
        parent_tags = get_parent_tags(tag)
        for parent_tag in parent_tags:
            if parent_tag == tag:
                continue
            if parent_tag in output_tags:
                print("Removing {}".format(parent_tag))
                output_tags.remove(parent_tag)
    return list(output_tags)

def get_parent_tags(tag):
    if tag.count('/') < 1:
        return []
    parent = re.match(r'(.*)/.*', tag)[1]
    return [tag] + get_parent_tags(parent)
