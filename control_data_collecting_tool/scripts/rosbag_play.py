import sqlite3
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message


ACCELERATION_TOPIC_NAME = '/localization/acceleration'
KINEMATIC_STATE_TOPIC_NAME = '/localization/kinematic_state'


class db3SingleTopic:
    def __init__(self, sqlite_file, topic_id):
        self.conn = sqlite3.connect(sqlite_file)
        self.c = self.conn.cursor()
        self.c.execute('SELECT * from messages WHERE topic_id = {}'.format(topic_id))
        self.is_open = True

    def __del__(self):
        self.conn.close()

    def fetch_row(self):
        return self.c.fetchone()
    

class db3Converter:

    def __init__(self, db3_file):
        self.db3_file = db3_file

        self.db3_dict = {}


    def __del__(self):
        for topic_db3 in self.db3_dict.values():
            del topic_db3["topic_db3"]


    def load_db3(self, topic_name):
        try:
            conn = sqlite3.connect(self.db3_file)
            c = conn.cursor()
            c.execute('SELECT * from({})'.format('topics'))
            topics = c.fetchall()

        except sqlite3.Error as e:
            return False
        
        topic_types = {topics[i][1]: {"topic_id":i+1,"topic_type":get_message(topics[i][2])} for i in range(len(topics))}

        target_topic_included = True
        if not topic_name in list(topic_types.keys()):
                target_topic_included = False
        conn.close()

        if not target_topic_included:
            return target_topic_included
        
        #
        topic_type = topic_types[topic_name]["topic_type"]
        topic_db3 = db3SingleTopic(self.db3_file, topic_types[topic_name]["topic_id"])
        self.db3_dict[topic_name] = {"topic_type":topic_type, "topic_db3":topic_db3}
        
        return True
    

    def read_msg(self, topic_name):
        msg = None
        row = self.db3_dict[topic_name]["topic_db3"].fetch_row()

        if row is None:
            return msg
        
        topic_type = self.db3_dict[topic_name]["topic_type"]
        msg = deserialize_message( row[3], topic_type )

        return msg
    