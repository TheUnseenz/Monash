--****PLEASE ENTER YOUR DETAILS BELOW****
--T3-rm-dm.sql

--Student ID:
--Student Name:

/* Comments for your marker:




*/

--(a) Create Sequences
-- Drop sequence for COMPETITOR if it exists
DROP SEQUENCE competitor_seq;

-- Create sequence for COMPETITOR
CREATE SEQUENCE competitor_seq
START WITH 100
INCREMENT BY 5;

-- Drop sequence for TEAM if it exists
DROP SEQUENCE team_seq;

-- Create sequence for TEAM
CREATE SEQUENCE team_seq
START WITH 100
INCREMENT BY 5;


-- (b) Record New Competitors, Team, and Entries (8 marks)
SET TRANSACTION NAME 'Register Keith and Jackson, form Super Runners';

-- Insert Keith Rose (Competitor)
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone)
VALUES (competitor_seq.NEXTVAL, 'Keith', 'Rose', 'M', TO_DATE('15-JAN-1990', 'DD-MON-YYYY'), 'keith.rose@monash.edu', 'Y', '0422141112');

-- Insert Jackson Bull (Competitor)
INSERT INTO COMPETITOR (comp_no, comp_fname, comp_lname, comp_gender, comp_dob, comp_email, comp_unistatus, comp_phone)
VALUES (competitor_seq.NEXTVAL, 'Jackson', 'Bull', 'M', TO_DATE('20-MAR-1992', 'DD-MON-YYYY'), 'jackson.bull@monash.edu', 'Y', '0422412524');

-- Insert Jackson's entry for 10 Km Run (Jackson is initially sole team member and leader)
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
VALUES (
    (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code JOIN CARNIVAL c ON e.carn_date = c.carn_date WHERE c.carn_name = 'RM Winter Series Caulfield 2025' AND et.eventtype_desc = '10 Km Run'),
    (SELECT NVL(MAX(entry_no), 0) + 1 FROM ENTRY WHERE event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code JOIN CARNIVAL c ON e.carn_date = c.carn_date WHERE c.carn_name = 'RM Winter Series Caulfield 2025' AND et.eventtype_desc = '10 Km Run')),
    TO_TIMESTAMP('09:00:00', 'HH24:MI:SS'), NULL, NULL,
    (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524'),
    NULL, -- Team ID will be updated after team is created
    (SELECT char_id FROM CHARITY WHERE char_name = 'RSPCA')
);

-- Insert Super Runners team, with Jackson's entry as the leader's entry
INSERT INTO TEAM (team_id, team_name, carn_date, event_id, entry_no)
VALUES (team_seq.NEXTVAL, 'Super Runners',
       (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'),
       (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code JOIN CARNIVAL c ON e.carn_date = c.carn_date WHERE c.carn_name = 'RM Winter Series Caulfield 2025' AND et.eventtype_desc = '10 Km Run'),
       (SELECT entry_no FROM ENTRY WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524')
                                   AND event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code JOIN CARNIVAL c ON e.carn_date = c.carn_date WHERE c.carn_name = 'RM Winter Series Caulfield 2025' AND et.eventtype_desc = '10 Km Run'))
);

-- Update Jackson's entry to associate with the new Super Runners team
UPDATE ENTRY
SET team_id = (SELECT team_id FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025'))
WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524')
  AND event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code JOIN CARNIVAL c ON e.carn_date = c.carn_date WHERE c.carn_name = 'RM Winter Series Caulfield 2025' AND et.eventtype_desc = '10 Km Run');

-- Insert Keith's entry for 10 Km Run and assign to Super Runners team
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
VALUES (
    (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code JOIN CARNIVAL c ON e.carn_date = c.carn_date WHERE c.carn_name = 'RM Winter Series Caulfield 2025' AND et.eventtype_desc = '10 Km Run'),
    (SELECT NVL(MAX(entry_no), 0) + 1 FROM ENTRY WHERE event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code JOIN CARNIVAL c ON e.carn_date = c.carn_date WHERE c.carn_name = 'RM Winter Series Caulfield 2025' AND et.eventtype_desc = '10 Km Run')),
    TO_TIMESTAMP('09:00:00', 'HH24:MI:SS'), NULL, NULL,
    (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422141112'),
    (SELECT team_id FROM TEAM WHERE team_name = 'Super Runners' AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025')),
    (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army')
);

COMMIT;


-- (c) Downgrade Event (4 marks)
SET TRANSACTION NAME 'Downgrade 10 Km Run to 5 Km Run';

-- Insert the new 5 Km Run event
INSERT INTO EVENT (event_id, carn_date, eventtype_code, event_starttime)
VALUES (
    (SELECT NVL(MAX(event_id), 0) + 1 FROM EVENT), -- Dynamically get next event_id
    TO_DATE('29-JUN-2025', 'DD-MON-YYYY'),
    (SELECT eventtype_code FROM EVENTTYPE WHERE eventtype_desc = '5 Km Run'),
    TO_DATE('08:00', 'HH24:MI')
);

-- Re-register entries for the new 5 Km Run event, nominating Salvation Army
-- This inserts new entries for the 5km run based on existing 10km run entries.
-- The entry_no is carried over as requested in the task.
-- This assumes the composite primary key (event_id, entry_no) will not conflict.
INSERT INTO ENTRY (event_id, entry_no, entry_starttime, entry_finishtime, entry_elapsedtime, comp_no, team_id, char_id)
SELECT
    (SELECT e_new.event_id FROM EVENT e_new JOIN EVENTTYPE et_new ON e_new.eventtype_code = et_new.eventtype_code WHERE e_new.carn_date = TO_DATE('29-JUN-2025', 'DD-MON-YYYY') AND et_new.eventtype_desc = '5 Km Run'), -- New 5km event_id
    e_old.entry_no, -- Carry over original entry_no as per requirement
    TO_TIMESTAMP('08:00:00', 'HH24:MI:SS'), -- New start time for 5K event
    NULL,
    NULL,
    e_old.comp_no,
    e_old.team_id,
    (SELECT char_id FROM CHARITY WHERE char_name = 'Salvation Army') -- New charity as per requirement
FROM
    ENTRY e_old
WHERE
    e_old.event_id = (SELECT e_10k.event_id FROM EVENT e_10k JOIN EVENTTYPE et_10k ON e_10k.eventtype_code = et_10k.eventtype_code WHERE e_10k.carn_date = TO_DATE('29-JUN-2025', 'DD-MON-YYYY') AND et_10k.eventtype_desc = '10 Km Run');

-- Delete the old 10 Km Run event entries
DELETE FROM ENTRY
WHERE event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code WHERE e.carn_date = TO_DATE('29-JUN-2025', 'DD-MON-YYYY') AND et.eventtype_desc = '10 Km Run');

-- Delete the 10 Km Run event itself
DELETE FROM EVENT
WHERE carn_date = TO_DATE('29-JUN-2025', 'DD-MON-YYYY')
  AND eventtype_code = (SELECT eventtype_code FROM EVENTTYPE WHERE eventtype_desc = '10 Km Run');

COMMIT;


-- (d) Keith Withdraws, Team Disbands (4 marks)
SET TRANSACTION NAME 'Keith withdraws, Super Runners disband';

-- Update Jackson's entry to remove him from the team
UPDATE ENTRY
SET team_id = NULL
WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422412524')
  AND event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code WHERE e.carn_date = TO_DATE('29-JUN-2025', 'DD-MON-YYYY') AND et.eventtype_desc = '5 Km Run');

-- Delete Keith's entry for the RM Winter Series Caulfield 2025 carnival 5 Km Run event
DELETE FROM ENTRY
WHERE comp_no = (SELECT comp_no FROM COMPETITOR WHERE comp_phone = '0422141112')
  AND event_id = (SELECT e.event_id FROM EVENT e JOIN EVENTTYPE et ON e.eventtype_code = et.eventtype_code WHERE e.carn_date = TO_DATE('29-JUN-2025', 'DD-MON-YYYY') AND et.eventtype_desc = '5 Km Run');

-- Delete the Super Runners team
DELETE FROM TEAM
WHERE team_name = 'Super Runners'
  AND carn_date = (SELECT carn_date FROM CARNIVAL WHERE carn_name = 'RM Winter Series Caulfield 2025');

COMMIT;