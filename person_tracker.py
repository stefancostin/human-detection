class PersonTracker:

    def __init__(self):
        self.tracked_persons = {}
        self.total_tracked_count = 0

    def add(self, person):
        self.total_tracked_count += 1
        self.tracked_persons[person.tag] = person

    def list(self):
        return self.tracked_persons.values()

    def remove(self, person_list):
        for person in person_list:
            del self.tracked_persons[person.tag]

    def size(self):
        return len(self.tracked_persons.keys())

    def total_count(self):
        return self.total_tracked_count
