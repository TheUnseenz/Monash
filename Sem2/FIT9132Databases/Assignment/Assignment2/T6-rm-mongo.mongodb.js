// *****PLEASE ENTER YOUR DETAILS BELOW*****
// T6-rm-mongo.mongodb.js

// Student ID: 27030768
// Student Name: Adrian Leong Tat Wei

// Comments for your marker:

// ===================================================================================
// DO NOT modify or remove any of the comments below (items marked with //)
// ===================================================================================

// Use (connect to) your database - you MUST update xyz001
// with your authcate username

use("atleo4");

// (b)
// PLEASE PLACE REQUIRED MONGODB COMMAND TO CREATE THE COLLECTION HERE
// YOU MAY PICK ANY COLLECTION NAME
// ENSURE that your query is formatted and has a semicolon
// (;) at the end of this answer

// Drop collection
db.teams.drop();

// Create collection and insert documents
db.teams.insertMany([
    {
        "_id": 1,
        "carn_name": "RM Spring Series Clayton 2024",
        "carn_date": "22-Sep-2024",
        "team_name": "Speedy Gazelles",
        "team_leader":
        {
            "name": "Alice Smith",
            "phone": "0412345678",
            "email": "alice.smith@monash.edu"
        },
        "team_no_of_members": 2,
        "team_members":
            [
                {
                    "competitor_name": "Bob Johnson",
                    "competitor_phone": "0412876543",
                    "event_type": "10 Km Run",
                    "entry_no": 2,
                    "starttime": "08:31:00",
                    "finishtime": "09:29:00",
                    "elapsedtime": "+00 00:58:00.000000"
                },
                {
                    "competitor_name": "Alice Smith",
                    "competitor_phone": "0412345678",
                    "event_type": "10 Km Run",
                    "entry_no": 1,
                    "starttime": "08:30:00",
                    "finishtime": "09:25:30",
                    "elapsedtime": "+00 00:55:30.000000"
                }
            ]
    },

    {
        "_id": 2,
        "carn_name": "RM Spring Series Caulfield 2024",
        "carn_date": "05-Oct-2024",
        "team_name": "Roadrunners",
        "team_leader":
        {
            "name": "Bob Johnson",
            "phone": "0412876543",
            "email": "bob.johnson@example.com"
        },
        "team_no_of_members": 2,
        "team_members":
            [
                {
                    "competitor_name": "Bob Johnson",
                    "competitor_phone": "0412876543",
                    "event_type": "10 Km Run",
                    "entry_no": 1,
                    "starttime": "08:30:00",
                    "finishtime": "09:28:00",
                    "elapsedtime": "+00 00:58:00.000000"
                },
                {
                    "competitor_name": "Alice Smith",
                    "competitor_phone": "0412345678",
                    "event_type": "10 Km Run",
                    "entry_no": 2,
                    "starttime": "08:35:00",
                    "finishtime": "09:30:15",
                    "elapsedtime": "+00 00:55:15.000000"
                }
            ]
    },

    {
        "_id": 3,
        "carn_name": "RM Winter Series Caulfield 2025",
        "carn_date": "29-Jun-2025",
        "team_name": "Trail Blazers",
        "team_leader":
        {
            "name": "Charlie Brown",
            "phone": "0412112233",
            "email": "charlie.brown@monash.edu"
        },
        "team_no_of_members": 3,
        "team_members":
            [
                {
                    "competitor_name": "Charlie Brown",
                    "competitor_phone": "0412112233",
                    "event_type": "5 Km Run",
                    "entry_no": 1,
                    "starttime": "08:30:00",
                    "finishtime": "09:28:00",
                    "elapsedtime": "+00 00:58:00.000000"
                },
                {
                    "competitor_name": "Ivy Lee",
                    "competitor_phone": "0412556677",
                    "event_type": "5 Km Run",
                    "entry_no": 4,
                    "starttime": "08:33:00",
                    "finishtime": "09:32:00",
                    "elapsedtime": "+00 00:59:00.000000"
                },
                {
                    "competitor_name": "Emily White",
                    "competitor_phone": "0412778899",
                    "event_type": "5 Km Run",
                    "entry_no": 3,
                    "starttime": "08:32:00",
                    "finishtime": "09:30:00",
                    "elapsedtime": "+00 00:58:00.000000"
                }
            ]
    },

    {
        "_id": 4,
        "carn_name": "RM Winter Series Caulfield 2025",
        "carn_date": "29-Jun-2025",
        "team_name": "Lone Wolves",
        "team_leader":
        {
            "name": "Diana Prince",
            "phone": "0412445566",
            "email": "diana.p@example.com"
        },
        "team_no_of_members": 2,
        "team_members":
            [
                {
                    "competitor_name": "Henry King",
                    "competitor_phone": "0412334455",
                    "event_type": "21.1 Km Half Marathon",
                    "entry_no": 2,
                    "starttime": "07:46:00",
                    "finishtime": "12:20:00",
                    "elapsedtime": "+00 04:34:00.000000"
                },
                {
                    "competitor_name": "Diana Prince",
                    "competitor_phone": "0412445566",
                    "event_type": "21.1 Km Half Marathon",
                    "entry_no": 1,
                    "starttime": "07:45:00",
                    "finishtime": "12:15:00",
                    "elapsedtime": "+00 04:30:00.000000"
                }
            ]
    },

    {
        "_id": 5,
        "carn_name": "RM Spring Series Clayton 2024",
        "carn_date": "22-Sep-2024",
        "team_name": "Forest Friends",
        "team_leader":
        {
            "name": "Emily White",
            "phone": "0412778899",
            "email": "emily.white@monash.edu"
        },
        "team_no_of_members": 2,
        "team_members":
            [
                {
                    "competitor_name": "Frank Green",
                    "competitor_phone": "0412998877",
                    "event_type": "5 Km Run",
                    "entry_no": 2,
                    "starttime": "08:01:00",
                    "finishtime": "08:26:00",
                    "elapsedtime": "+00 00:25:00.000000"
                },
                {
                    "competitor_name": "Emily White",
                    "competitor_phone": "0412778899",
                    "event_type": "5 Km Run",
                    "entry_no": 1,
                    "starttime": "08:00:00",
                    "finishtime": "08:25:00",
                    "elapsedtime": "+00 00:25:00.000000"
                }
            ]
    },

    {
        "_id": 6,
        "carn_name": "RM Spring Series Clayton 2024",
        "carn_date": "22-Sep-2024",
        "team_name": "Night Owls",
        "team_leader":
        {
            "name": "Alice Smith",
            "phone": "0412345678",
            "email": "alice.smith@monash.edu"
        },
        "team_no_of_members": 2,
        "team_members":
            [
                {
                    "competitor_name": "Bob Johnson",
                    "competitor_phone": "0412876543",
                    "event_type": "5 Km Run",
                    "entry_no": 4,
                    "starttime": "08:03:00",
                    "finishtime": "08:28:00",
                    "elapsedtime": "+00 00:25:00.000000"
                },
                {
                    "competitor_name": "Alice Smith",
                    "competitor_phone": "0412345678",
                    "event_type": "5 Km Run",
                    "entry_no": 3,
                    "starttime": "08:02:00",
                    "finishtime": "-",
                    "elapsedtime": "-"
                }
            ]
    }
]);

// List all documents you added
db.teams.find({});


// (c)
// PLEASE PLACE REQUIRED MONGODB COMMAND/S FOR THIS PART HERE
// ENSURE that your query is formatted and has a semicolon
// (;) at the end of this answer

db.teams.find(
    {
        "team_members.event_type": { $in: ["5 Km Run", "10 Km Run"] }
    },
    {
        "carn_date": 1,
        "carn_name": 1,
        "team_name": 1,
        "team_leader.name": 1,
        "_id": 0
    }
);


// (d)
// PLEASE PLACE REQUIRED MONGODB COMMAND/S FOR THIS PART HERE
// ENSURE that your query is formatted and has a semicolon
// (;) at the end of this answer

// (i) Add new team
db.teams.insertOne(
    {
        "_id": 101, // Manually decided _id
        "carn_name": "RM Winter Series Caulfield 2025",
        "carn_date": "29-Jun-2025",
        "team_name": "The Great Runners",
        "team_leader": {
            "name": "Jackson Bull",
            "phone": "0422412524",
            "email": "jackson.bull@example.com"
        },
        "team_no_of_members": 1,
        "team_members": [
            {
                "competitor_name": "Jackson Bull",
                "competitor_phone": "0422412524",
                "event_type": "5 Km Run",
                "entry_no": 42,
                "starttime": "08:45:00",
                "finishtime": "-",
                "elapsedtime": "-"
            }
        ]
    }
);


// Illustrate/confirm changes made
db.teams.find(
    {
        "team_name": "The Great Runners",
        "carn_name": "RM Winter Series Caulfield 2025"
    }
);


// (ii) Add new team member
db.teams.updateOne(
    {
        "team_name": "The Great Runners",
        "carn_name": "RM Winter Series Caulfield 2025"
    },
    {
        $inc: { "team_no_of_members": 1 }, // Increment member count
        $push: {
            "team_members": {
                "competitor_name": "Steve Bull",
                "competitor_phone": "0422251427",
                "event_type": "10 Km Run",
                "entry_no": 43, // Next entry number for this event within the team context
                "starttime": "08:30:00",
                "finishtime": "-",
                "elapsedtime": "-"
            }
        }
    }
);


// Illustrate/confirm changes made
db.teams.find(
    {
        "team_name": "The Great Runners",
        "carn_name": "RM Winter Series Caulfield 2025"
    }
);

