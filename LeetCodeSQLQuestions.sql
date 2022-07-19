# 175. Combine Two Tables
select 
    p.firstName,
    p.lastName,
    a.city,
    a.state
from Person p
left join Address a 
on p.personId = a.personId;



# 176. Second Highest Salary
# solution 1
select max(salary) as SecondHighestSalary
from employee
where salary < (select max(salary) from employee);

# solution 2,3
# ifnull(distinct salary,null)
# distinct can not inside the ifnull
# also you have to use the bracket
select (
    ifnull(
	(select distinct salary
    from Employee
    order by salary desc 
    limit 1 offset 1)
    ,null) 
) as SecondHighestSalary;

# use it as temp table would work too
select 
   ( select distinct salary
    from Employee
    order by salary desc 
    limit 1 offset 1 
) as SecondHighestSalary;


# solution 4,5
with temp as 
(
	select distinct salary
    from employee
    order by salary desc
    limit 2
)
select salary as secondhighestsalary
from temp
order by salary asc
limit 1;

with temp as 
(
	select salary,
			dense_rank() over(order by salary desc) as denserank
	from employee
)
select salary as secondhighestsalary
from temp
where denserank = 2;



# 177. Nth Highest Salary
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
SET N=N-1;
  RETURN (
      # Write your MySQL query statement below.
        select distinct salary 
        from employee 
        order by salary desc
        limit 1 offset N
  );
END

# 178. Rank Scores
select 
    score,
    dense_rank() over(order by score desc) as "rank"
from Scores;



# 180. Consecutive Numbers
select distinct a.num as ConsecutiveNums
from Logs a
inner join Logs b 
on a.id = b.id+1 and a.num = b.num
inner join Logs c
on a.id = c.id+2 and a.num = c.num;



# 181. Employees Earning More Than Their Managers
select 
    e.name as "Employee"
from Employee e
join Employee m on m.id = e.managerId
where e.salary > m.salary;



# 182. Duplicate Emails
select email
from Person
group by email
having count(email) > 1;



# 183. Customers Who Never Order
# solution 1
select c.name as "Customers"
from Customers c
where c.id not in 
(
    select customerId from orders
);
# solution 2
select c.name as 'customers'
from customer c 
left join orders o
on c.id = o.customerId
where customerId is null; 



# 184. Department Highest Salary
select 
    d.name as 'Department',
    e.name as 'Employee',
    e.salary as 'Salary'
from Employee e 
join Department d
on e.departmentId = d.id
where (e.departmentId,e.salary) in 
(
    select departmentId, max(salary)
    from Employee 
    group by departmentId);



# 185. Department Top Three Salaries
with temp as
(
    select 
    d.name as 'Department',
    e.name as 'Employee',
    e.salary as 'Salary',
    dense_rank() over(partition by e.departmentId order by e.salary desc) as "rank"
from Employee e
join Department d on e.departmentId = d.id )
select 
    Department,
    Employee,
    Salary
from temp
where temp.rank<4
order by Department, Salary;



# 196. Delete Duplicate Emails
# solution 1
delete p2 
from Person p1 
join Person p2 on p1.email = p2.email
where p1.id <p2.id;

# solution 2
delete from Person where Id not in
(
    select*from(
    select 
        min(id) 
    from Person 
    group by email) alias
);



# 197. Rising Temperature
# join on datediff(today, yesterday)
select w2.id
from Weather w1 
join Weather w2 on datediff(w2.recordDate,w1.recordDate) =1
where w1.temperature < w2.temperature;



# 262. Trips and Users
# if(condition,result,result)
select
    request_at as Day,
    Round(sum(if(status="completed",0,1))/count(request_at),2)as 'Cancellation Rate'
from 
    Trips
where client_id in 
    (select users_id
    from Users
    where banned = 'No' and role = 'client')
and driver_id in
    (select users_id
    from Users
    where banned = 'No' and role = 'driver')
and request_at between '2013-10-01' and '2013-10-03'
group by request_at;



# 511. Game Play Analysis I
select player_id,
		min(event_date) as first_login
from Activity
group by player_id;



# 512. Game Play Analysis II
# solution 1
select place_id,
		device_id
from Activity
group by player_id
having min(event_date);

# solution 2
select distinct player_id, device_id
from Activity
where(player_id,event_date) in 
(
	select player_id, min(event_date)
    from Activity
    group by player_id
);


# 534. Game Play Analysis III
# window function
select player_id,
	event_date,
    sum(games_palyed) over(partition by player_id order by event_date asc) as games_played_so_far
from Activity;



# 550. Game Play Analysis IV
# solution 1
select round(count(distinct b.player_id)/count(distinct a.player_id),2) as fraction
from (
	select player_id,
		min(event_date) as event_date
	from Activity
    group by player_id ) a 
left join Acitivty b
on a.player_id = b.player_id and a.event_date+1 = b.event_date;

# solution 2
# round (sum (case when then else end))
select round(sum(case when a.event_date = b.first_event+1 then 1 else 0 end)/count(distinct a.player_id),2) as fraction
from Activity a
join (
	select player_id, min(event_date) as first_event
    from Activity
    group by player_id
) b
on a.player_id = b.player_id;



# 569. Median Employee Salary 
select t1.Id, t1.Company, t1.Salary
from Employee as t1 join Employee as t2
on t1.Company = t2.Company
group by t1.Id
having abs(sum(case when t2.Salary<t1.Salary then 1
					when t2.Salary>t1.Salary then -1
                    when t2.Salary=t1.Salary and t2.Id<t1.Id then 1
                    when t2.Salary=t1.Salary and t2.Id>t1.Id then -1 else 0 end)) <=1
order by t1.Company, t1.Salary, t1.Id;



# 570. Managers with at Least 5 Direct Reports
select Name
from Employee
where Id in (
	select ManagerId    
	from Employee
    where count(distinct Id) >=5 );
    


# 571. Find Median Given Frequency of Numbers
SELECT avg(t3.Number) as median
FROM Numbers as t3
JOIN
    (SELECT t1.Number,
        abs(SUM(CASE WHEN t1.Number>t2.Number THEN t2.Frequency ELSE 0 END) -
            SUM(CASE WHEN t1.Number<t2.Number THEN t2.Frequency ELSE 0 END)) AS count_diff
    FROM numbers AS t1, numbers AS t2
    GROUP BY t1.Number) AS t4
ON t3.Number = t4.Number
WHERE t3.Frequency>=t4.count_diff;



# 574. Winning Candidate
select Name 
from Candidate
where id = (Select CandidateId
			from Vote
			group by CandidateId
			order by count(1) desc
			limit 1);
            


# 577. Employee Bonus 
select e.name, b.bonus
from Employee e
left join Bonus b on e.empId = b.empId
where b.bonus < 1000;



# 578. Get Highest Answer Rate Question
# solution 1
select question_id as survey_log
from (
	select question_id,
    sum(case when action='show' then 1 else 0 end) as show_count,
    sum(case when action='answer' then 1 else 0 end) as answer_count
    from survey_log
    group by question_id
) alias
order by answer_count/show_cout desc limit 1;

# solution 2
select question_id as survey_log
from (
	select question_id,
    sum(if(action='show', 1, 0) as show_count,
    sum(if(action='answer', 1, 0) as answer_count
    from survey_log
    group by question_id
) alias
order by answer_count/show_cout desc limit 1;



# 579. Find Cumulative Salary of an Employee
select Id, Month, sum(Salary) over(partition by Id order by Month asc) as Salary
from Employee
group by Id, Month
having Month < max(month) and count(Id)<4
order by Id asc, Month desc;



# 580. Count Student Number in Departments
# solution 1
select 
	d.dept_name,
	count(s.student_id) over(partition by s.dept_id) as student_number
from student s
right join department d on d.dept_id = s.dept_id
group by d.dept_name
order by student_number desc, d.dept_name asc;

# solution 2
select 
	dept_name,
	sum(case when student_id is null then 0 else 1 end) as student_number
from department d
left join student s
on d.dept_id = s.dept_id
group by d.dept_name
order by student_number desc, dept_name;



# 584. Find Customer Referee
select name 
from Customer
where referee_id <> 2 or referee_id is null;

select name 
from Customer
where referee_id != 2 or isnull(referee_id) ; -- not equal doesn't take care the null value



# 585. Investments in 2016
# solution 1
select round(sum(TIV_2016),2) as TIV_2016
from 
(
	select TIV_2015,
		TIV_2016,
        concat(lat,',',lon)
	from insurance
    group by TIV_2015,lat,lon
    having count(TIV_2015)>=1 and count(concat(lat,',',lon))=1
) alias;

# solution 2
select round(sum(TIV_2016),2) as TIV_2016
from insurance
where concat(lat,',',lon) in 
(
	select concat(lat,',',lon) 
    from insurance
    group by lat, lon
    having count(1) =1
)
and TIV_2015 in 
(
	select TIV_2015
    from insurance 
    group by TIV_2015
    having count(1)>1
);



# 586. Customer Placing the Largest Number of Orders
# order by count() desc limit 1 so you can get the largest
select customer_number
from Orders
group by customer_number
order by count(order_number) desc
limit 1;



# 595. Big Countries
# it's 'or' not 'and'
select name, population, area
from world
where area > 300000 or population > 25000000;



# 596. Classes More Than 5 Students
select class
from Courses
group by class
having count(student) >= 5;



# 597. Friend Requests I: Overall Acceptance Rate 
# ifnull()
select ifnull(round(count(accept)/count(sender),2),0.00) as accept_rate
from (
	select 
		count(distinct sender_id, send_to_id) as send,
        count(distinct requester_id, accepter_id) as accept
	from request_accepted r
	right join friend_request f
	on r.sender_id = f.requester_id and r.send_to_id = f.accepter_id
);

select ifnull(round(accepts/requests,2),0.00) as accept_rate
from (
	select count(distinct sender_id, send_to_id) as requests from friend_request
) t1, 
(
	select count(distinct requester_id, accepter_id) as accepts from request_accepted
) t2;



# 601. Human Traffic of Stadium
# hard question
# very tricky
select distinct t1.*
from stadium t1, stadium t2, stadium t3
where t1.people >= 100 and t2.people >= 100 and t3.people >= 100
and
(
    (t1.id - t2.id = 1 and t1.id - t3.id = 2 and t2.id - t3.id =1)  -- t1, t2, t3
    or
    (t2.id - t1.id = 1 and t2.id - t3.id = 2 and t1.id - t3.id =1) -- t2, t1, t3
    or
    (t3.id - t2.id = 1 and t2.id - t1.id =1 and t3.id - t1.id = 2) -- t3, t2, t1
)
order by t1.id;

# better solution 
# you have to use the self join
select distinct s1.* 
from stadium s1 join stadium s2 join stadium s3
on (s1.id = s2.id - 1 and s1.id = s3.id - 2) OR -- s1.id < s2.id < s3.id 
(s1.id = s2.id + 1 and s1.id = s3.id - 1) OR  -- s2.id < s1.id < s3.id but remember we only want s1.*
(s1.id = s2.id + 1 and s1.id = s3.id + 2) -- s1.id > s2.id > s3.id 
where s1.people >= 100 and s2.people >= 100 and s3.people >= 100
order by visit_date;



# 602. Friend Requests II: Who Has the Most Friends
# solution 1
select requester_id as id,
	max(f_num) as num
from(
	select *, 
	count(requester_id) over(partition by accepter_id) as req_num,
	count(accepter_id) over(partition by requester_id) as acc_num,
	sum(count(requester_id) over(partition by accepter_id)+count(accepter_id) over(partition by requester_id)) as f_num
	from request_accepted
) alias
# solution 2
select t.id, sum(t.num) as num
from 
(
	(select requester_id as id, count(1) as num
    from request_accepted
    group by requester_id)
    union all
    (select accepte_id as id, count(1) as num
    from request_accepted
    group by accepter_id)
) as t
group by t.id
order by num desc
limit 1;



# 603. Consecutive Available Seats
select distinct t1.seat_id
from cinema as t1 
join cinema as t2
on abs(t1.seat_id - t2.seat_id) =1
where t1.free='1' and t2.free='1'
order by t1.seat_id;



# 607.Sales Person
# this solution would not work b/c there are sales_id work with both
select distinct s.name
from salesperson s
join company c on s.sales_id = o.sales_id
join orders o on c.com_id = o.com_id
where c.com_id != '1'; # c.name != 'RED'

# pretty good solution
select name
from salesperson
where name not in 
	(
		select distinct s.name
        from salesperson s, orders o, company c
        where c.name ='RED'
        and s.sales_id = o.sales_id
        and o.com_id = c.com_id
	) ;
    
SELECT
    s.name
FROM
    salesperson s
WHERE
    s.sales_id NOT IN (SELECT
            o.sales_id
        FROM
            orders o
                LEFT JOIN
            company c ON o.com_id = c.com_id
        WHERE
            c.name = 'RED');



# 608. Tree Node
# better option 1
# Inner: select p_id from Tree where p_id in (select id from Tree where p_id is not null)
# p_id = null is not working
# p_is is null would work
select id,
case when p_id is null then 'Root'
when p_id is not null and id in (select p_id from Tree where p_id is not null) then 'Inner'
when id not in (select p_id from Tree where p_id is not null) then 'Leaf' end as type -- or else 'Leaf' would work
from Tree
order by id asc;

# better option 2 
# if(,,if(,,))
# isnull()
select id,
if (isnull(p_id), 'Root',
if( id in (select p_id from Tree where p_id is not null), 'Inner','Leaf')) as type
from Tree
order by id asc;

## Basic Ideas: LEFT JOIN
# In tree, each node can only one parent or no parent
## | id | p_id | id (child) |
## |----+------+------------|
## |  1 | null |          1 |
## |  1 | null |          2 |
## |  2 |    1 |          4 |
## |  2 |    1 |          5 |
## |  3 |    1 |       null |
## |  4 |    2 |       null |
## |  5 |    2 |       null |

SELECT t1.id,
    CASE
        WHEN ISNULL(t1.p_id) THEN 'Root'
        WHEN ISNULL(MAX(t2.id)) THEN 'Leaf'
        ELSE 'Inner'
    END AS Type
FROM tree AS t1 LEFT JOIN tree AS t2
ON t1.id = t2.p_id
GROUP BY t1.id, t1.p_id;



# 610. Triangle Judgement
select *,
	(case when x+y>z and y+z>x and x+z>y then 'Yes' else 'No' end) as triangle
from triangle;



# 612. Shortest Distance in a Plane
select round(min(sqrt((t1.x-t2.x)*(t1.x-t2.x)+(t1.y-t2.y)*(t1.y-t2.y))),2) as shortest
from point_2d as t1, point_2d as t2
where t1.x!=t2.x or t1.y!=t2.y;



# 613. Shortest Distance in a Line
select min(p2.x-p1.x) as shortest
from point as p1, point as p2
where p1.x < p2.x;

select t1.x-t2.x as shortest
from point as t1 join point as t2
where t1.x > t2.x
order by t1.x-t2.x asc
limit 1;



# 614. Second Degree Follower 
select followee as follower,
	count(distinct follower) over(partition by followee) as num
from follow
where followee in 
	(select distinct follower from follow)

## Explain the business logic
##   A follows B. Then A is follwer, B is followee
## What are second degree followers?
##   A follows B, and B follows C.
##   Then A is the second degree followers of C

SELECT f1.follower, COUNT(DISTINCT f2.follower) AS num
FROM follow AS f1 JOIN follow AS f2
ON f1.follower = f2.followee
GROUP BY f1.follower;



# 615. Average Salary: Departments VS Company
select pay_month,
	department_id,
    (case when dep_tot > com_tot then "higher"
		when dep_tot < com_tot then "lower"
        else "same" end)as comparison
from 
(
	select month(s.pay_date)as pay_month,
			s.department_id,
            avg(amount) over(partition by month(s.pay_date)) as com_tot,
            avg(amount) over(partition by month(s.pay_date), s.department_id) as dep_tot
    from salary s 
    join employee e on s.employee_id = e.employee_id 
) alias;



# 618. Students Report By Geography 
select 
	case when continent = 'America' then name else null end as America,
	case when continent = 'Asia' then name else null end as Asia,
	case when continent = 'Europe' then name else null end as Europe
from student
order by America, Asia, Europe;



# 619. Biggest Single Number
select ifnull(max(num),null) as num
from number
where count(num) = 1;



# 620. Not Boring Movies
select * 
from
    Cinema
where description <> 'boring' and id%2=1
order by rating desc;



# 626. Exchange Seats
select
    if(id<(select max(id) from Seat), if(id%2=0,id-1,id+1), if(id%2=0,id-1,id)) as id, 
    student
from Seat
order by id asc;



# 627. Swap Salary
update Salary set sex = (case when sex = 'm' then 'f' else 'm' end);

update Salary set sex = if(sex='m', 'f','m');



# 1045. Customers Who Bought All Products
select customer_id
from Customer
group by customer_id
having count(distinct product_key) = (select count(*) from Product)
order by customer_id;



# 1050. Actors and Directors Who Cooperated At Least Three Times
select actor_id, director_id
from ActorDirector
group by actoer_id,director_id
having count(timestamp)>=3;

select actor_id, director_id
from 
(
    select distinct actor_id, 
    director_id,
    count(timestamp) over(partition by actor_id, director_id) as cnt
    from ActorDirector
) temp
where cnt >= 3;



# 1068. Product Sales Analysis I 
select p.product_name,
	s.year,
    s.price
from Sales s
join Product p on s.product_id = p.product_id
order by s.year;



# 1069. Product Sales Analysis II 
select product_id,
	sum(quantity)over(partition by product_id)as total_quantity
from Sales
order by product_id;



# 1070. Product Sales Analysis III
select product_id,
	year,
    quantity,
    price
from Sales
where (product_id,year) in 
	(select product_id,
		min(year) as first_year
	from Sales
	group by product_id);
    
    

# 1075. Project Employees I
select project_id,
	round(avg(exerience_year) over(partition by project_id),2) as average_years
from Project p
join Employee e
on p.employee_id = e.employee_id



#1076. Project Employees II 
select project_id
from Project
where count(employee_id) = (select max(count(employee_id) over(partition by project_id)) from Project)



# 1077. Project Employees III
select project_id,
	employee_id
from (
select project_id,
	employee_id,
    rank() over(partition by project id order by experience_year desc) as rnk
from Project p
join Employee e
on p.employee_id = e.employee_id
) alias
where rnk = 1
order by project_id, employee_id;

select p.project_id,
	e.employee_id
from Project p
left join Employee e on p.employee_id = e.employee_id
where (p.project_id,e.experience_years) in (
select p.project_id,
max(e.experience_years)
from Project p join Employee e on p.employee_id = e.employee_id group by p.project_id);



# 1082. Sales Analysis I
select seller_id
from 
(
	select s.seller_id, 
    sum(s.price) over(partition by seller_id) as sel_tot,
    from Sales
    order by sum(s.price) desc
    limit 1
) alias;

SELECT seller_id
FROM Sales
GROUP BY seller_id
HAVING SUM(price) = (SELECT SUM(price)
                    FROM Sales
                    GROUP BY seller_id
                    ORDER BY SUM(price) DESC
                    LIMIT 1) ;



# 1083. Sales Analysis II
select distinct buyer_id
from Sales s 
left join Product p 
on s.product_id = p.product_id
where p.product_name = 'S8' and s.buyer_id not in 
(
	select s.buyer_id 
    from Sales s 
    left join Product p on s.product_id = p.product_id
    where p.product_name ='iPhone'
);



# 1084. Sales Analysis III
# the trick part is where sale_date condition and product_id not in 
select distinct p.product_id, -- still need distinct here b/c sales doesn't have primary key
p.product_name
from Product p
join Sales s
on s.product_id = p.product_id
where s.sale_date >= '2019-01-01' and s.sale_date <= '2019-03-31'  -- use the date wisely
and s.product_id not in(
    select distinct product_id
    from Sales -- no need to join the product table
    where sale_date > '2019-03-31' or sale_date < '2019-01-01');  -- use the date wisely

#     
select product.product_id, product_name
from product join sales on product.product_id = sales.product_id
group by product_id having min(sale_date) >= "2019-01-01" and max(sale_date) <= "2019-03-31";

select product.product_id, product_name
from product join sales on product.product_id = sales.product_id
group by product_id, product_name
having SUM(sale_date BETWEEN "2019-01-01" AND "2019-03-31") > 0
and SUM(sale_date > "2019-03-31" or sale_date < "2019-01-01") = 0;
    


# 1097. Game Play Analysis V 
select install_dt,
	count(a1.player_id) over(partition by install_dt) as installs,
    round((count(a2.player_id)over(partition by install_dt))/(count(a1.player_id) over(partition by install_dt)),2) as Day1_retention
from(
select a1.player_id, a2.player_id
min(a1.event_date) over(partition by a1.player_id) as install_dt,
from Activity as a1 
left join Activity as a2
on a2.event_date = a1.event_date+1 and a2.player_id = a1.palyer_id
) alias;

SELECT
    install_dt,
    COUNT(player_id) installs,
    ROUND(COUNT(retention)/COUNT(player_id),2) Day1_retention  # the number of record on the next day / the total number of id on the day = retention rate
FROM
    (
    SELECT a.player_id, a.install_dt, b.event_date retention # id, the record of the first installation day and next day
    FROM
        (SELECT player_id, MIN(event_date) install_dt   # subquery 1 take the first installation of date of each id
        FROM Activity
        GROUP BY player_id) a LEFT JOIN Activity b ON   # sq1 left join the original table, find the login status the next day after the first installation
            a.player_id = b.player_id AND
            a.install_dt + 1=b.event_date
    ) AS tmp
GROUP BY
    install_dt;
    
    

# 1098. Unpopular Books
select b.book_id, b.name
from Books b
join (
	select book_id, 
	sum(quantity) over(partition by book_id) as tot
	from Orders
	where dispatch_date between '2018-06-23' and '2019-06-23') temp 
on temp.order_id = b.order_id
where book_id in (select book_id
				from Books
				where available_from < date_add('2019-06-23', interval - 1 month)) 
and tot < 10 or tot is null;

SELECT
    b.book_id, b.name
FROM
    Books b LEFT JOIN (                    -- subquery calculates last year's sales
        SELECT book_id, SUM(quantity) nsold
        FROM Orders
        WHERE dispatch_date BETWEEN '2018-06-23' AND '2019-06-23'
        GROUP BY book_id
    ) o
    ON b.book_id = o.book_id
WHERE
    (o.nsold < 10 OR o.nsold IS NULL) AND           -- Sales less than 10 or no sales
    DATEDIFF('2019-06-23', b.available_from) > 30;   -- Not a new book within 1 month



# 1107. New Users Daily Count 
select 
	login_date, 
    count(user_id) over(partition by login_date) as user_count
from(
select user_id, min(activity_date) over(partition by user_id) as login_date
from Traffic
where activity ="login" and datediff('2019-06-30',activity_date) <= 90) alias;

#Solution- 1:
SELECT login_date, COUNT(user_id) AS user_count
FROM (SELECT user_id, MIN(activity_date) AS login_date
    FROM Traffic
    WHERE activity = 'login'
    GROUP BY user_id) AS t
WHERE login_date >= DATE_ADD('2019-06-30', INTERVAL -90 DAY) AND login_date <= '2019-06-30'
GROUP BY login_date;

#Solution- 2:
SELECT login_date, COUNT(user_id) user_count
FROM
    (SELECT user_id, MIN(activity_date) as login_date
    FROM Traffic
    WHERE activity='login'
    GROUP BY user_id) as t
WHERE DATEDIFF('2019-06-30', login_date) <= 90
GROUP BY login_date;



# 1112. Highest Grade For Each Student
select student_id, min(course_id) over(partition by student_id) as course_id, grade
from 
	(select 
		student_id, course_id
		max(grade) over(partition by student_id order by course_id asc) as grade
	from Enrollments) temp
order by student_id asc;

SELECT student_id, MIN(course_id) course_id, grade
FROM Enrollments
WHERE (student_id, grade) IN
        (SELECT student_id, MAX(grade)
        FROM Enrollments
        GROUP BY student_id)
GROUP BY student_id
ORDER BY student_id;



# 1113.Reported Posts
select extra as report_reason, 
	report_count
from (
	select 
		distinct post_id, 
        extra, 
        count(distinct post_id) over(partition by extra) as report_count
    from Actions
	where action = 'report' and action_date = date_sub('2019-07-05', interval 1 day)
) temp ;

SELECT extra report_reason, COUNT(DISTINCT post_id) report_count
FROM
    (SELECT post_id, extra
    FROM Actions
    WHERE action_date = DATE_SUB('2019-07-05', INTERVAL 1 DAY) AND
          action = 'report') AS tmp
GROUP BY extra;



# 1126. Active Businesses
select 
	distinct business_id
from
(	
	select business_id,
		event_type
    from Events
    group by event_type
	having occurences > avg(occurences)
) alias
group by business_id
having count(event_type) > 1 ;

SELECT business_id
FROM (SELECT a.business_id, a.event_type, a.occurences, b.event_avg  -- sub 2
      FROM Events a LEFT JOIN
        (SELECT event_type, AVG(occurences) event_avg   -- sub 1
         FROM Events
         GROUP BY event_type) b ON
      a.event_type = b.event_type) tmp
WHERE occurences > event_avg
GROUP BY business_id
HAVING COUNT(event_type) > 1;



# 1127. User Purchase Platform
select spend_date,
	platform,
    coalesce(total_amount, 0) as total_amount,
    coalesce(total_users, 0) as total_users
from (
select spend_date, 
case when plat_num = 2 then 'both'
	when platform = 'mobile' and plat_num = 1 then 'mobile'
	when platform = 'desktop' and plat_num = 1 then 'desktop' end as platform,
    sum(amount) over(partition by spend_date,platform) as total_amount,
    count(user_id) over(partition by spend_date,platform) as total_users
from (
select 
	spend_date, 
    user_id,
    platform,
    count(platform) over(partition by spend_date,user_id) as plat_num
from Spending ) alias1
) alias2;

SELECT aa.spend_date,
       aa.platform,
       COALESCE(bb.total_amount, 0) total_amount,
       COALESCE(bb.total_users,0) total_users
FROM
    (SELECT DISTINCT(spend_date), a.platform   -- table aa
    FROM Spending JOIN
        (SELECT 'desktop' AS platform UNION
        SELECT 'mobile' AS platform UNION
        SELECT 'both' AS platform
        ) a
    ) aa
    LEFT JOIN
    (SELECT spend_date,                      -- table bb
            platform,
            SUM(amount) total_amount,
            COUNT(user_id) total_users
    FROM
        (SELECT spend_date,
                user_id,
                (CASE COUNT(DISTINCT platform)
                    WHEN 1 THEN platform
                    WHEN 2 THEN 'both'
                    END) platform,
                SUM(amount) amount
        FROM Spending
        GROUP BY spend_date, user_id
        ) b
    GROUP BY spend_date, platform
    ) bb
    ON aa.platform = bb.platform AND
    aa.spend_date = bb.spend_date；


# 1132. Reported Posts II
select 
	round(avg(cnt/num_re) over(partition by action_date) *100,2) as average_daily_percent
from
(
	select a.action_date,
			b.remove_date,
		count(a.post_id) over(partition by a.action_date) as cnt,
        count(b.remove_date) over(partition by a.action_date) as num_re
    from Action as a
    left join Removals as r
    on a.action_id = r.action_id and a.post_id = r.post_id
    where a.action = 'report' and a.extra = 'spam'
) temp;

select 
	round(avg(percent_of_remove) over(partition by action_date) *100,2) as average_daily_percent
from
(
	select a.action_date,
			b.remove_date,
		(count(distinct a.post_id) over(partition by a.action_date) / count(distinct b.remove_date) over(partition by a.action_date)) as percent_of_remove
    from Action as a
    left join Removals as r
    on a.post_id = r.post_id
    where a.action = 'report' and a.extra = 'spam'
) temp;



# 1141. User Activity for the Past 30 Days I
# interval not intervel typo
# make sure the group by is correct
# and group by should be from select
# extra user_id in group by is no good
# also no primary key for this table so distinct
# MySQL doesn't yet support '<window function>(DISTINCT ..)'
select activity_date as day,
count(distinct user_id) as active_users -- count(distinct ) is okay, not okay with over( partition by)
from Activity
where activity_date <= '2019-07-27' 
and activity_date > date_sub('2019-07-27', interval 30 day)
group by activity_date;



# 1142. User Activity for the Past 30 Days II
# ifnull(,0)
select
 ifnull(round(avg(session_id_per_user),2),0) as average_sessions_per_user
from (
	select
		count(distinct session_id) over(partition by user_id) as session_id_per_user
	from Activity
	where activity_date <= '2019-07-27' and activity_date >= date_sub('2019-07-27', interval 30 day)
) temp;



# 1148. Article Views I
select distinct author_id as id
from Views
where author_id = viewer_id
order by author_id asc; -- order by id would work too!



# 1149. Article Views II
select distinct viewer_id as id
from
(
select distinct viewer_id,
count(author_id) over(partition by view_date,viewer_id) as cnt_view
	from Views
) temp
where cnt_view > 1
order by viewer_id asc;

select distinct viewer_id as id
from Views
group by view_date,viewer_id
having count(distinct author_id) >1
order by viewer_id asc;



# 1158. Market Analysis I 
# better option 1
# The ON clause defines the relationship between the tables.
# The WHERE clause describes which rows you are interested in.

# Many times you can swap them and still get the same result,
# however this is not always the case with a LEFT OUTER JOIN.
# If the ON clause fails you still get a row with columns from the left table 
# but with nulls in the columns from the right table.
# If the WHERE clause fails you won't get that row at all.

# WHERE is a part of the SELECT query as a whole, ON is a part of each individual join.
# ON can only refer to the fields of previously used tables.
# When there is no actual match against a record in the left table, 
# LEFT JOIN returns one record from the right table with all fields set to NULLS. 
# WHERE clause then evaluates and filter this.
select u.user_id as buyer_id,
u.join_date,
ifnull(count(o.order_id),0) as orders_in_2019
from Users u
left outer join Orders o
on u.user_id = o.buyer_id and year(o.order_date) =2019
group by u.user_id;

# better option 2
select distinct u.user_id as buyer_id,
u.join_date,
coalesce(count(o.order_id) over(partition by u.user_id),0) as orders_in_2019
from Users u
left outer join Orders o
on u.user_id = o.buyer_id and year(o.order_date) =2019;

# option 3
select u.user_id as buyer_id,
u.join_date,
coalesce(sum(if(year(order_date) = '2019',1,0)),0) as orders_in_2019
from Users u
left join Orders o
on o.buyer_id = u.user_id
group by user_id

# extract('year' from order_date)
# coalesce(,0)
select
	u.user_id as buyer_id,
	join_date,
	coalesce(t.orders_in_2019,0) as orders_in_2019
from Users as u
left join 
(
	select o.buyer_id as buyer_id,
		u.join_date as join_date,
		count(o.buyer_id)over(partition by order_date) as orders_in_2019
	from Users as u
	join Orders as o
	on u.user_id = o.buyer_id
	where extract('year' from order_date) = '2019'
) as t
on u.user_id = t.buyer_id

SELECT user_id AS buyer_id, join_date, coalesce(a.orders_in_2019,0)
FROM users
LEFT JOIN
(
SELECT buyer_id, coalesce(count(*), 0) AS orders_in_2019
FROM orders o
JOIN users u
ON u.user_id = o.buyer_id
WHERE extract('year' FROM order_date) = 2019
GROUP BY buyer_id) a
ON users.user_id = a.buyer_id



# 1159. Market Analysis II
#  limit 1 offset 1

select user_id as seller_id,
	ifnull(case when item_id = sec_id then 'yes'
		else 'no' end, 'no') as '2nd_item_fav_brand'
from Users u
left join
(
	select u.user_id,
		i.item_id,
		t.sec_id
	from Users u
	join Item i on i.item_brand = u.favorite_brand
	join 
	(	select
			seller_id,
			item_id as sec_id
		from 
		(
			select *,
				count(item_id) over(partition by seller_id) as cnt
				from Orders
		) t1
		where cnt >= 2
		order by seller_id, order_date asc
		limit 1 offset 1
	) t2
	on t2.seller_id = u.user_id 
) t3
on t3.user_id = u.user_id;


#Solution- 1:
SELECT user_id AS seller_id,
       IF(ISNULL(item_brand), "no", "yes") AS 2nd_item_fav_brand
FROM Users LEFT JOIN
(SELECT seller_id, item_brand
FROM Orders INNER JOIN Items
ON Orders.item_id = Items.item_id
WHERE (seller_id, order_date) IN
(SELECT seller_id, MIN(order_date) AS order_date
 FROM Orders
 WHERE (seller_id, order_date) NOT IN
 (SELECT seller_id, MIN(order_date) FROM Orders GROUP BY seller_id)
GROUP BY seller_id)
 ) AS t
ON Users.user_id = t.seller_id and favorite_brand = item_brand;

#Solution- 2:
# RANK() OVER(PARTITION BY seller_id ORDER BY order_date) AS rk
WITH t1 AS(
SELECT user_id,
CASE WHEN favorite_brand = item_brand THEN "yes"
ELSE "no"
END AS 2nd_item_fav_brand
FROM users u LEFT JOIN
(SELECT o.item_id, seller_id, item_brand, RANK() OVER(PARTITION BY seller_id ORDER BY order_date) AS rk
FROM orders o join items i
USING (item_id)) a
ON u.user_id = a.seller_id
WHERE a.rk = 2)

SELECT u.user_id AS seller_id, COALESCE(2nd_item_fav_brand,"no") AS 2nd_item_fav_brand
FROM users u LEFT JOIN t1
USING(user_id);



# 1164. Product Price at a Given Date
# IF(ISNULL(item_brand), "no", "yes")
select 
distinct product_id,
if(isnull(new_price),10,new_price) as price 
from Products p
left join
(select product_id, 
new_price, 
max(change_date) over(partition by product_id)
from Products
where change_date <= '2019-08-16') temp
on p.product_id = temp.product_id;



# 1173. Immediate Food Delivery I 
with t1 as
(
	select 
		deliver_id, 
		count(*) as imm
    from delivery
    where order_date = customer_pref_delivery_date
)
select round(t1.imm/count(d.deliver_id)*100,2) as immediate_percentage
from t1, Delivery d;

#solution 1
select 
round(sum(case when order_date=customer_pref_delivery_date then 1 else 0 end)/count(1)*100,2) as immediate_percentage
from Delivery;

#solution 2
select 
round(avg(case when order_date=customer_pref_delivery_date then 1 else 0 end)*100,2) as immediate_percentage
from Delivery;



# 1174. Immediate Food Delivery II 
select round(sum(case when first_order = customer_pref_delivery_date then 1 else 0 end)/count(customer_id)*100,2) as immediate_percentage
from  
(
	select
	distinct customer_id,
	min(order_date) as first_order,
    min(customer_pref_delivery_date) as customer_pref_delivery_date
    from Delivery
    group by customer_id
) temp;

select 
round(avg(case when first_order=customer_pref_delivery_date then 1 else 0 end)*100,2) as immediate_percentage
from 
(
	select *,
    rank() over(partition by customer_id order by order_date asc) as rnk
    from Delivery
) temp
where temp.rnk = 1;

select round(sum(case when order_date = customer_pref_delivery_date then 1 else 0 end)/count(customer_id)*100,2) as immediate_percentage
from Delivery
where (customer_id, order_date) in
	(select customer_id, min(order_date)
		from Delivery
        group by customer_id);



# 1179. Reformat Department Table
# the trick part is sum() group by id, bc there's different department id
select id,
sum(case when month = 'Jan' then revenue else null end) as Jan_Revenue,
sum(case when month = 'Feb' then revenue else null end) as Feb_Revenue,
sum(case when month = 'Mar' then revenue else null end) as Mar_Revenue,
sum(case when month = 'Apr' then revenue else null end) as Apr_Revenue,
sum(case when month = 'May' then revenue else null end) as May_Revenue,
sum(case when month = 'Jun' then revenue else null end) as Jun_Revenue,
sum(case when month = 'Jul' then revenue else null end) as Jul_Revenue,
sum(case when month = 'Aug' then revenue else null end) as Aug_Revenue,
sum(case when month = 'Sep' then revenue else null end) as Sep_Revenue,
sum(case when month = 'Oct' then revenue else null end) as Oct_Revenue,
sum(case when month = 'Nov' then revenue else null end) as Nov_Revenue,
sum(case when month = 'Dec' then revenue else null end) as Dec_Revenue
from Department 
group by id;

select distinct id,
sum(case when month = 'Jan' then revenue else null end) over(partition by id) as Jan_Revenue,
sum(case when month = 'Feb' then revenue else null end) over(partition by id)  as Feb_Revenue,
sum(case when month = 'Mar' then revenue else null end) over(partition by id)  as Mar_Revenue,
sum(case when month = 'Apr' then revenue else null end) over(partition by id)  as Apr_Revenue,
sum(case when month = 'May' then revenue else null end) over(partition by id)  as May_Revenue,
sum(case when month = 'Jun' then revenue else null end) over(partition by id)  as Jun_Revenue,
sum(case when month = 'Jul' then revenue else null end) over(partition by id)  as Jul_Revenue,
sum(case when month = 'Aug' then revenue else null end) over(partition by id)  as Aug_Revenue,
sum(case when month = 'Sep' then revenue else null end) over(partition by id)  as Sep_Revenue,
sum(case when month = 'Oct' then revenue else null end) over(partition by id)  as Oct_Revenue,
sum(case when month = 'Nov' then revenue else null end) over(partition by id)  as Nov_Revenue,
sum(case when month = 'Dec' then revenue else null end) over(partition by id)  as Dec_Revenue
from Department;

SELECT id,
	sum(IF(month='Jan', revenue, NULL)) AS Jan_Revenue,
	sum(IF(month='Feb', revenue, NULL)) AS Feb_Revenue,
	sum(IF(month='Mar', revenue, NULL)) AS Mar_Revenue,
	sum(IF(month='Apr', revenue, NULL)) AS Apr_Revenue,
	sum(IF(month='May', revenue, NULL)) AS May_Revenue,
	sum(IF(month='Jun', revenue, NULL)) AS Jun_Revenue,
	sum(IF(month='Jul', revenue, NULL)) AS Jul_Revenue,
	sum(IF(month='Aug', revenue, NULL)) AS Aug_Revenue,
	sum(IF(month='Sep', revenue, NULL)) AS Sep_Revenue,
	sum(IF(month='Oct', revenue, NULL)) AS Oct_Revenue,
	sum(IF(month='Nov', revenue, NULL)) AS Nov_Revenue,
	sum(IF(month='Dec', revenue, NULL)) AS Dec_Revenue
FROM Department
group by id;



# 1193. Monthly Transactions I
select date_format(trans_date,'%Y-%m') as month,
	country,
    count(id) over(partition by country, month(trans_date)) as trans_count,
    sum(amount) over(partition by country, month(trans_date)) as trans_total_amount,
    sum(if(state="approved",1,0))over(partition by country, month(trans_date)) as approved_count,
	sum(if(state="approved",amount,0))over(partition by country, month(trans_date)) as approved_total_amount
from Transactions;

WITH t1 AS(
SELECT DATE_FORMAT(trans_date,'%Y-%m') AS month, country, COUNT(state) AS trans_count, sum(amount) AS trans_total_amount
FROM transactions
GROUP BY country, month(trans_date)),

t2 AS (
SELECT DATE_FORMAT(trans_date,'%Y-%m') AS month, country, COUNT(state) AS approved_count, sum(amount) AS approved_total_amount
FROM transactions
WHERE state = 'approved'
GROUP BY country, month(trans_date))

SELECT t1.month, t1.country, COALESCE(t1.trans_count,0) AS trans_count, COALESCE(t2.approved_count,0) AS approved_count, COALESCE(t1.trans_total_amount,0) AS trans_total_amount, COALESCE(t2.approved_total_amount,0) AS approved_total_amount
FROM t1 LEFT JOIN t2
ON t1.country = t2.country and t1.month = t2.month;



# 1194. Tournament Winners 
# row_number()
# union all
# coalesce()
select group_id,
	player_id 
from
(
	select p.player_id,
	p.group_id,
	sum(t.score) over(partition by t.player_id) as tot,
    row_number() over(partition by group_id order by group_id, tot desc) as rnk
	from 
	(	select first_player as player_id,
		sum(first_score) over(partition by first_player) as score
		from Matches
		union all
		select second_player as player_id,
		sum(second_score) over(partition by second_player) as score
		from Matches
	) t
	join Player p
	on p.player_id = t.player_id
) tmp
where rnk = 1;


WITH t1 AS(
SELECT first_player, SUM(first_score) AS total
FROM
(SELECT first_player, first_score
FROM matches
UNION ALL
SELECT second_player, second_score
FROM matches) a
GROUP BY 1),

t2 AS(
SELECT *, COALESCE(total,0) AS score
FROM players p LEFT JOIN t1
ON p.player_id = t1.first_player)

SELECT group_id, player_id
FROM
(SELECT *, ROW_NUMBER() OVER(PARTITION BY group_id ORDER BY group_id, score DESC) AS rn
FROM t2) b
WHERE b.rn = 1;



# 1204. Last Person to Fit in the Elevator
select person_name
from
(
	select person_name,
		sum(weight) over(order by turn) as cnt_1000
	from Queue
) temp
where cnt_1000 <=1000
order by cnt_1000 desc
limit 1;


WITH t1 AS
(
SELECT *,
SUM(weight) OVER(ORDER BY turn) AS cum_weight
FROM queue
ORDER BY turn)

SELECT t1.person_name
FROM t1
WHERE turn = (SELECT MAX(turn) FROM t1 WHERE t1.cum_weight<=1000);



# 1205. Monthly Transactions II 
select month(t.trans_date) as month,
t.country,
count(if(t.state='approved',1,0)) over(partition by month(t.trans_date),t.country) as approved_count,
sum(if(t.state='approved',t.amount,0)) over(partition by month(t.trans_date),t.country) as approved_amount,
temp.chargeback_count,
temp.chargeback_amount
from Transactions t
join 
(
	select c.trans_id,
	month(c.trans_date) as ch_month,
    t.country,
    count(c.trans_id) over(partition by month(c.trans_date),t.country) as chargeback_count,
    sum(t.amount)over(partition by month(c.trans_date),t.country) as chargeback_amount
	from Chargebacks c
	join Transactions t
	on c.trans_id = t.id
) temp
on temp.ch_month = t.month and temp.country = t.country
order by month;

#Solution 1:
# extract('month' from trans_date)
# group by 1,2,3
# coalesce(,0)
WITH t1 AS
(SELECT country, extract('month' FROM trans_date), state, COUNT(*) AS approved_count, SUM(amount) AS approved_amount
FROM transactions
WHERE state = 'approved'
GROUP BY 1, 2, 3),
t2 AS(
SELECT t.country, extract('month' FROM c.trans_date), SUM(amount) AS chargeback_amount, COUNT(*) AS chargeback_count
FROM chargebacks c LEFT JOIN transactions t
ON trans_id = id
GROUP BY t.country, extract('month' FROM c.trans_date)),

t3 AS(
SELECT t2.date_part, t2.country, COALESCE(approved_count,0) AS approved_count, COALESCE(approved_amount,0) AS approved_amount, COALESCE(chargeback_count,0) AS chargeback_count, COALESCE(chargeback_amount,0) AS chargeback_amount
FROM t2 LEFT JOIN t1
ON t2.date_part = t1.date_part AND t2.country = t1.country),

t4 AS(
SELECT t1.date_part, t1.country, COALESCE(approved_count,0) AS approved_count, COALESCE(approved_amount,0) AS approved_amount, COALESCE(chargeback_count,0) AS chargeback_count, COALESCE(chargeback_amount,0) AS chargeback_amount
FROM t2 RIGHT JOIN t1
ON t2.date_part = t1.date_part AND t2.country = t1.country)

SELECT *
FROM t3
UNION
SELECT *
FROM t4;

#Solution 2:
SELECT month, country,
    SUM(CASE WHEN type='approved' THEN 1 ELSE 0 END) AS approved_count,
    SUM(CASE WHEN type='approved' THEN amount ELSE 0 END) AS approved_amount,
    SUM(CASE WHEN type='chargeback' THEN 1 ELSE 0 END) AS chargeback_count,
    SUM(CASE WHEN type='chargeback' THEN amount ELSE 0 END) AS chargeback_amount
FROM (
    (
    SELECT left(t.trans_date, 7) AS month, t.country, amount,'approved' AS type
    FROM Transactions AS t
    WHERE state='approved'
    )
    UNION ALL (
    SELECT left(c.trans_date, 7) AS month, t.country, amount,'chargeback' AS type
    FROM Transactions AS t JOIN Chargebacks AS c
    ON t.id = c.trans_id
    )
) AS tt
GROUP BY tt.month, tt.country;

#Solution 3:
SELECT month, country,
    SUM(CASE WHEN type='approved' THEN count ELSE 0 END) AS approved_count,
    SUM(CASE WHEN type='approved' THEN amount ELSE 0 END) AS approved_amount,
    SUM(CASE WHEN type='chargeback' THEN count ELSE 0 END) AS chargeback_count,
    SUM(CASE WHEN type='chargeback' THEN amount ELSE 0 END) AS chargeback_amount
FROM (
    (
    SELECT LEFT(t.trans_date, 7) AS month, t.country,
    COUNT(1) AS count, SUM(amount) AS amount,'approved' AS type
    FROM Transactions AS t LEFT JOIN Chargebacks AS c
    ON t.id = c.trans_id
    WHERE state='approved'
    GROUP BY LEFT(t.trans_date, 7), t.country
    )
    union (
    SELECT LEFT(c.trans_date, 7) AS month, t.country,
        COUNT(1) AS count, SUM(amount) AS amount,'chargeback' AS type
    FROM Transactions AS t JOIN Chargebacks AS c
    ON t.id = c.trans_id
    GROUP BY LEFT(c.trans_date, 7), t.country
    )
) AS tt
GROUP BY tt.month, tt.country;



# 1211. Queries Quality and Percentage
select distinct query_name,
round(coalesce(quality,0),2) as quality,
round(coalesce(less_3/tot*100,0),2) as poor_query_percentage
from 
(
	select 
		distinct query_name,
		avg(rating/position) over(partition by query_name) as quality,
        sum(if(rating<3,1,0)) over(partition by query_name) as less_3,
        count(rating) over(partition by query_name) as tot
	from Queries
    
) temp;

select distinct query_name,
round(avg(rating/position) ,2) as quality,
round(sum(if(rating<3,1,0)) /count(rating) *100,2) as poor_query_percentage
from Queries
group by query_name;

select distinct query_name,
round(avg(rating/position) ,2) as quality,
round(sum(case when rating<3 then 1 else 0 end) /count(rating) *100,2) as poor_query_percentage
from Queries
group by query_name;



# 1212. Team Scores in Football Tournament
with temp as
(
	select *,
    (case when host_goals > guest_goals then 3 
		when host_goals = guest_goals then 1
        else 0 end) as host_point,
	(case when host_goals > guest_goals then 0 
		when host_goals = guest_goals then 1
        else 3 end) as guest_point
	from Matches
)

select team_id, team_name,
sum(points) over(partition by team_id) as num_points
from 
(
	select 
		t1.host_team as team_id, t1.host_point as points, team_name
	from temp t1
    join Teams t on t.team_id = t1.host_team 
	union all
	select
		t1.guest_team as team_id, t1.guest_point as points,team_name
	from temp t1
    join Teams t on t.team_id = t1.guest_team 
) temp2
order by num_points desc;

#Solution 1:
# use '+' 
SELECT Teams.team_id, Teams.team_name,
    SUM(CASE WHEN team_id=host_team AND host_goals>guest_goals THEN 3 ELSE 0 END) +
    SUM(CASE WHEN team_id=host_team AND host_goals=guest_goals THEN 1 ELSE 0 END) +
    SUM(CASE WHEN team_id=guest_team AND host_goals<guest_goals THEN 3 ELSE 0 END) +
    SUM(CASE WHEN team_id=guest_team AND host_goals=guest_goals THEN 1 ELSE 0 END) AS num_points
FROM Teams LEFT JOIN Matches
ON Teams.team_id = Matches.host_team OR Teams.team_id = Matches.guest_team
GROUP BY Teams.team_id
ORDER BY num_points DESC, Teams.team_id ASC;



# 1225. Report Contiguous Dates
select period_state,
	case when next_day > date_add(dates, interval + 1 day) then next_day 
	when dates = min(dates) then dates end as start_date,
    case when next_day > date_add(dates, interval + 1 day) then dates 
	when dates = max(dates) then dates 
    when next_day = null then dates end as end_date
from(
select 'failed' as period_state,
fail_date as dates, 
lead(success_date,1) over (order by success_date) next_day
from Failed
where fail_date >= '2019-01-01' and fail_date <= '2019-12-31'
union all
select 'successded' as period_state,
success_date as dates, 
lead(success_date,1) over (order by success_date) next_day
from Succeeded
where success_date >= '2019-01-01' and success_date <= '2019-12-31') temp;

## Solution 2:
## First generate a list of dates
##   succeeded 2019-01-01
##   succeeded 2019-01-02
##   ...
##   failed 2019-01-04
##   ...
## Add group id for contiguous ranges
## Notice: dates themselves are contiguous
##
SELECT period_state, MIN(date) AS start_date, MAX(date) AS end_date
FROM (
    SELECT period_state, date,
         @rank := CASE WHEN @prev = period_state THEN @rank ELSE @rank+1 END AS rank,
         @prev := period_state AS prev
    FROM (
        SELECT 'failed' AS period_state, fail_date AS date
        FROM Failed
        WHERE fail_date BETWEEN '2019-01-01' AND '2019-12-31'
        UNION
        SELECT 'succeeded' AS period_state, success_date AS date
        FROM Succeeded
        WHERE success_date BETWEEN '2019-01-01' AND '2019-12-31') AS t,
        (SELECT @rank:=0, @prev:='') AS rows
    ORDER BY date ASC) AS tt
GROUP BY rank
ORDER BY rank;



# 1241. Number of Comments per Post
select post_id, coalesce(number_of_comments,0) as number_of_comments
from
(
	select distinct parent_id as post_id,
	count(distinct sub_id) over(partition by parent_id) as number_of_comments
	from Submissions
	where parent_id in (
						select distinct sub_id
						from Submission
						where parent_id is Null)
) temp;

SELECT a.sub_id AS post_id, coalesce(b.number_of_comments,0) AS number_of_comments
FROM(
SELECT DISTINCT sub_id FROM submissions WHERE parent_id IS NULL) a
LEFT JOIN(
SELECT parent_id, count(DISTINCT(sub_id)) AS number_of_comments
FROM submissions
GROUP BY parent_id
HAVING parent_id = any(SELECT sub_id from submissions WHERE parent_id IS NULL)) b
ON a.sub_id = b.parent_id
ORDER BY post_id;



# 1251. Average Selling Price
select u.product_id, 
	round(sum(p.price*u.units) over(partition by product_id)/sum(u.units) over(partition by product_id),2) as average_price
from Prices p
right join UnitsSold u 
on p.product_id = u.product_id
and u.purchase_date < p.end_date and u.purchase_date > p.start_date;

select u.product_id, 
	round(sum(p.price*u.units) over(partition by p.product_id)/sum(u.units) over(partition by product_id),2) as average_price
from Prices p
inner join UnitsSold u 
on p.product_id = u.product_id
and u.purchase_date between p.end_date and p.start_date
group by p.product_id;



# 1264. Page Recommendations
# union remove duplicate records
# union all including the duplicates 
select distinct page_id as recommended_page
from Likes
where user_id in (
					select distinct user1_id
					from Friendship
					where user2_id = 1
					union
					select distinct user2_id
					from Friendship
					where user1_id = 1)
and page_id not in (select page_id from Likes where user_id = 1 );



# 1270. All People Report to the Given Manager 

#Solution 1:
## t3: directly report to employee_id 1
## t2: directly report to t3
## t1: directly report to t2
SELECT t1.employee_id
FROM Employees AS t1 
INNER JOIN Employees AS t2
ON t1.manager_id = t2.employee_id
JOIN Employees AS t3
ON t2.manager_id = t3.employee_id
WHERE t3.manager_id = 1 AND t1.employee_id != 1;

#Solution 2:
# UNION 1, 2, 3
SELECT distinct employee_id
FROM (
		SELECT employee_id
		FROM Employees
		WHERE manager_id IN
						(SELECT employee_id
						FROM Employees
						WHERE manager_id IN
										(SELECT employee_id
										FROM Employees
										WHERE manager_id = 1))
		UNION
				SELECT employee_id
				FROM Employees
				WHERE manager_id IN
								(SELECT employee_id
								FROM Employees
								WHERE manager_id = 1)
		UNION
				SELECT employee_id
				FROM Employees
				WHERE manager_id = 1) AS t
WHERE employee_id != 1;

#Solution 3:
# WHERE manager_id = any (SELECT employee_id
SELECT employee_id
FROM employees
WHERE manager_id = 1 AND employee_id != 1
UNION
SELECT employee_id
FROM employees
WHERE manager_id = any (SELECT employee_id
FROM employees
WHERE manager_id = 1 AND employee_id != 1)
UNION
SELECT employee_id
FROM employees
WHERE manager_id = any (SELECT employee_id
FROM employees
WHERE manager_id = any (SELECT employee_id
FROM employees
WHERE manager_id = 1 AND employee_id != 1));

 

# 1280. Students and Examinations
select s.student_id, 
	s.student_name, 
    e.subject_name, 
count(e.student_id) over(partition by e.subject_name) as attended_exams
from Students s
join Examinations e on e.subject_id = s.student_id
order by s.student_id, e.subject_name;



# 1285. Find the Start and End Number of Continuous Ranges
# lag(log_id,1) over(order by log_id) as previous_id
select case when min(log_id) = log_id then log_id 
		when log_id+1 > next_id then next_id
		end as start_id,
        case when max(log_id) = log_id then log_id
        when log_id+1 > next_id then log_id
        when log_id = null then log_id
        end as end_id
from(
	select log_id,
    lead(log_id,1) over(order by log_id) as next_id
    from Logs
	) temp;

#Solution 1: using log_id - ROW_NUMBER()
SELECT MIN(log_id) AS start_id, MAX(log_id) AS end_id
FROM(
SELECT log_id, log_id-ROW_NUMBER() OVER (ORDER BY log_id) AS rk
FROM logs) a
GROUP BY rk

#Solution 2: Add temporary columns of rank and prev
SELECT MIN(log_id) AS START_ID, MAX(log_id) AS END_ID
FROM (SELECT log_id,
        @rank := CASE WHEN @prev = log_id-1 THEN @rank ELSE @rank+1 END AS rank,
        @prev := log_id AS prev
    FROM Logs,
       (SELECT @rank:=0, @prev:=-1) AS rows) AS tt
GROUP BY rank
ORDER BY START_ID

# Solution 3: Find the starting and ending sequences, then merge two AS one table
## find the starting sequence: 1, 7, 10
## find the ending sequence: 3, 8, 10
## merge them AS one table
SELECT start_id, MIN(end_id) AS end_id
FROM (SELECT t1.log_id AS start_id
    FROM logs AS t1 LEFT JOIN logs AS t2
    ON t1.log_id-1 = t2.log_id
    WHERE t2.log_id IS NULL) tt_start join
    (SELECT t1.log_id AS end_id
    FROM logs AS t1 LEFT JOIN logs AS t2
    ON t1.log_id+1 = t2.log_id
    WHERE t2.log_id IS NULL) tt_end
WHERE start_id<=end_id
GROUP BY start_id;



# 1294. Weather Type in Each Country
select 
	c.country_name,
    case when avg(w.weather_state) <= 15 then 'Cold'
    when avg(w.weather_state)  >= 25 then 'Hot'
    else 'Warm' end as weather_type
from Countries c
join Weather w
on w.country_id = c.country_id
group by c.country_id
having c.day between '2019-11-01' and '2019-11-30';

# month(day) = 11
SELECT country_name, CASE WHEN AVG(weather_state) <= 15 THEN "Cold"
                          WHEN AVG(weather_state) >= 25 THEN "Hot"
                          ELSE "Warm" END AS weather_type
FROM Countries INNER JOIN Weather
ON Countries.country_id = Weather.country_id
WHERE MONTH(day) = 11
GROUP BY country_name;



# 1303. Find the Team Size
select employee_id,
count(employee_id) over(partition by team_id) as team_size
from Employee;

# solution 2: 
SELECT employee_id, b.team_size
FROM employee e
JOIN
(
SELECT team_id, count(team_id) AS team_size
FROM employee
GROUP BY team_id) b
ON e.team_id = b.team_id;



# 1308. Running Total for Different Genders
select gender,
	day,
    sum(score_points) over(partition by gender order by day) as total
from Scores
order by gender, day;


#Solution 1: group by 1,2;
SELECT gender, day,
SUM(score_points) OVER(PARTITION BY gender ORDER BY day) AS total
FROM scores
GROUP BY 1,2
ORDER BY 1,2;

#Solution 2:
SELECT t1.gender, t1.day, SUM(t2.score_points) AS total
FROM Scores AS t1 JOIN Scores AS t2
ON t1.gender = t2.gender
AND t1.day>=t2.day
GROUP BY t1.gender, t1.day;



# 1321. Restaurant Growth
# order by visited_on rows 6 preceding
select
	visited_on,
    sum(amount) over(order by visited_on rows 6 preceding) as amount,
	round(avg(amount) over(order by visited_on rows 6 preceding),2) as average_amount
from
(
	select visited_on, sum(amount) over(partition by visited_on) as amount
    from customer
    order by visited_on
) temp
order by visited_on offset 5 rows;



# 1322. Ads Performance
select ad_id,
	round(coalesce(sum(if(action="Clicked", 1,0))/sum(if(action="Ignored", 0,1))*100,0),2) as ctr
from Ads
group by ad_id
order by ctr desc;

# Solution 1:
SELECT ad_id,
    (CASE WHEN clicks+views = 0 THEN 0 ELSE ROUND(clicks/(clicks+views)*100, 2) END) AS ctr
FROM
    (SELECT ad_id,
        SUM(CASE WHEN action='Clicked' THEN 1 ELSE 0 END) AS clicks,
        SUM(CASE WHEN action='Viewed' THEN 1 ELSE 0 END) AS views
    FROM Ads
    GROUP BY ad_id) AS t
ORDER BY ctr DESC, ad_id ASC;

# Solution 2:
WITH t1 AS(
SELECT ad_id, SUM(CASE WHEN action in ('Clicked') THEN 1 ELSE 0 END) AS clicked
FROM ads
GROUP BY ad_id
)

, t2 AS
(
SELECT ad_id AS ad, SUM(CASE WHEN action in ('Clicked','Viewed') THEN 1 ELSE 0 END) AS total
FROM ads
GROUP BY ad_id
)

SELECT a.ad_id, coalesce(round((clicked +0.0)/nullif((total +0.0),0)*100,2),0) AS ctr
FROM
(
select *
FROM t1 JOIN t2
ON t1.ad_id = t2.ad) a
ORDER BY ctr DESC, ad_id;



# 1327. List the Products Ordered in a Period
select product_name, unit
from(
select p.product_name, sum(unit) over(partition by product_id) as unit
from Orders o
join Products p
on o.product_id = p.product_id
where month(o.order_date) = 2 and year(o.order_date) = 2020) temp
where unit >= 100;



# 1336. Number of Transactions per Visit

with RECURSIVE T1 AS (
	select visit_date,
		coalesce(num_visits,0) as num_visits,
		coalesce(num_tran,0) as num_trans
	from(
		select v.user_id, v.visit_date,
			t.user_id, t.transaction_date,
			count(v.visit_date) over(partition by v.user_id, v.visit_date order by v.visit_date) as num_visits,
			count(t.transaction_date) over(partition by t.user_id,t.transaction_date order by t.transaction_date) as num_trans
		from Visits v
		left join Transactions t
		on v.user_id = t.user_id and v.visit_date = t.transaction_date
		) temp2
	),
    t2 as 
    (
		select select max(num_trans) as trans
        from t1
        union all
        select trans-1
        from t2
        where trans >=1 
	)
select trans as transactions_count,
coalesce(visits_count,0) as visits_count
from t2 left join ( 
					select num_trans as transactions_count, 
                    coalesce(count(*),0) as visits_count
                    from t1
                    group by 1
                    order by 1) as a
on a.transactions_count = t2. trans
order by 1;



# 1341. Movie Rating 
# order by with count()
# order by with avg()
select u.name as results
from Users u
join Movie_Rating mr on m.user_id = u.user_id
group by u.user_id
order by count(mr.movie_id) desc, u.name limit 1
union all
(select m.title as results
from Movies m
join Movie_Rating mr on m.movie_id = mr.movie_id
where month(mr.created_at) = 2 and year(mr.created_at) = 2020
group by mr.movie_id
order by avg(mr.rating) desc, m.title limit 1);
# where rating = max(rating);



# 1350. Students With Invalid Departments 
select id, name
from Students
where department_id not in (select id from Departments);

# where row is null
select s.id, s.name
from students s left join departments d
on s.department_id = d.id
where d.name is null;



# 1355. Activity Participants
select activity
from Friends 
where activity not in (
	select count(id) over(partition by activity) as rnk
    from Friends
    order by rnk limi 1
    union
    select count(id) over(partition by activity) as rnk
    from Friends
    order by rnk desc limi 1
);

with cte as (
	select count(*) as cnt, activity
    from Friends 
    Group by activity
)
select activity from cte
where cnt not in (
				select max(cnt) from cte
                union all
                select min(cnt) from cte
				);



# 1364. Number of Trusted Contacts of a Customer
with cte as (
select user_id, 
	count(contact_email) over(partition by user_id) as contacts_cnt,
    count(trusted_contacts) over(partition by user_id) as trusted_contacts_cnt
	from(
		select user_id, 
			contact_name, 
			contact_email,
			case when customer_name=null and email=null then 0 else 1 end as trusted_contacts
		from 
		(
			select con.user_id, con.contact_name, con.contact_email, cu.customer_name, cu.email 
			from Customers cu
			right join Contacts con
			on cu.customer_name = con.contact_name and cu.email = con.contact_email
		) t1
	) t2
)

select i.invoice_id, 
cu.customer_name,
i.price,
cte.contacts_cnt,
cte.trusted_contacts_cnt
from Customers cu
join Invoices i on i.user_id = cu.customer_id
join cte on cte.user_id = cu.customer_id
order by i.invoice_id;

select i.invoice_id, 
cu.customer_name,
i.price,
count(con.contact_email) over(partition by con.user_id) as contacts_cnt,
sum(case when con.contact_name,con.contact_email in 
			(select customer_name, email from Customer) then 1 else 0 end) 
over(partition by con.user_id) as trusted_contacts_cnt
from Customers cu
join Invoices i on i.user_id = cu.customer_id
join Contacts on con.user_id = cu.customer_id
order by i.invoice_id;

# use sum(case when in () then else end) 
SELECT invoice_id, customer_name, price,
 COUNT(Contacts.user_id) AS contacts_cnt,
 SUM(CASE WHEN Contacts.contact_name IN
    (SELECT customer_name FROM Customers)
    THEN 1 ELSE 0 END) AS trusted_contacts_cnt
FROM Invoices INNER JOIN Customers ON Invoices.user_id = Customers.customer_id
LEFT JOIN Contacts ON Customers.customer_id = Contacts.user_id
GROUP BY Invoices.invoice_id, customer_name
ORDER BY Invoices.invoice_id;
	


# 1369. Get the Second Most Recent Activity
select  username, activity, startDate, endDate
from (
select * , count(startDate) over(partition by username) as act_cnt,
 row_number() over(partition by username order by endDate desc) as rnk 
from UserActivity ) temp
where (act_cnt > 1 and rnk =2) or (act_cnt =1 and rnk =1 );  

(SELECT *
FROM UserActivity
GROUP BY username
HAVING count(1) = 1)
UNION
(SELECT a.*
FROM UserActivity AS a LEFT JOIN UserActivity AS b
on a.username = b.username AND a.endDate<b.endDate
GROUP BY a.username, a.endDate
HAVING count(b.endDate) = 1);



# 1378. Replace Employee ID With The Unique Identifier
select eu.unique_id,
	e.name,
from Employees e
left join EmployeeUNI eu 
on eu.id = e.id;



# 1384. Total Sales Amount by Year

SELECT
    b.product_id,
    a.product_name,
    a.yr AS report_year,
    CASE
        WHEN YEAR(b.period_start)=YEAR(b.period_end) AND a.yr=YEAR(b.period_start) THEN DATEDIFF(b.period_end,b.period_start)+1
        WHEN a.yr=YEAR(b.period_start) THEN DATEDIFF(DATE_FORMAT(b.period_start,'%Y-12-31'),b.period_start)+1
        WHEN a.yr=YEAR(b.period_end) THEN DAYOFYEAR(b.period_end)
        WHEN a.yr>YEAR(b.period_start) AND a.yr<YEAR(b.period_end) THEN 365
        ELSE 0
    END * average_daily_sales AS total_amount
FROM
    (SELECT product_id,product_name,'2018' AS yr FROM Product
    UNION
    SELECT product_id,product_name,'2019' AS yr FROM Product
    UNION
    SELECT product_id,product_name,'2020' AS yr FROM Product) a
JOIN Sales b
ON a.product_id=b.product_id
HAVING total_amount > 0
ORDER BY b.product_id,a.yr;



# 1393. Capital Gain/Loss
# operation_day is one of the primary key so no need for distinct or order by
select stock_name,
sum(if(operation = 'Buy',-price, price)) as capital_gain_loss 
from Stocks
group by stock_name;

# distinct + window function
select distinct stock_name,
sum(if(operation = 'Buy',-(price),+(price))) over(partition by stock_name)as capital_gain_loss
from Stocks;

# operation_day is one of the primary key so no need for distinct or order by
select stock_name,
sum(case when operation = 'Buy' then -price else price end) as capital_gain_loss 
from Stocks
group by stock_name;

# distinct + window function
select distinct stock_name,
sum(case when operation = 'Buy' then -price else price end) over(partition by stock_name)as capital_gain_loss
from Stocks;



# 1398. Customers Who Bought Products A and B but Not C
SELECT *
FROM Customers
WHERE customer_id IN
    (SELECT DISTINCT customer_id
     FROM Orders
     WHERE product_name = 'A'
    ) AND
    customer_id IN
    (SELECT DISTINCT customer_id
     FROM Orders
     WHERE product_name = 'B'
    ) AND
    customer_id NOT IN
    (SELECT DISTINCT customer_id
     FROM Orders
     WHERE product_name = 'C'
    )
ORDER BY customer_id;

#Solution 1:
WITH t1 AS
(
SELECT customer_id
FROM orders
WHERE product_name = 'B' AND
customer_id IN (SELECT customer_id
FROM orders
WHERE product_name = 'A'))

SELECT t1.customer_id, c.customer_name
FROM t1 JOIN customers c
ON t1.customer_id = c.customer_id
WHERE t1.customer_id != all(SELECT customer_id
FROM orders
WHERE product_name = 'C');

#Solution 2:
SELECT Customers.*
FROM (
    SELECT customer_id,
     sum(CASE WHEN product_name = 'A' THEN 1 ELSE 0 END) AS product_a,
     sum(CASE WHEN product_name = 'B' THEN 1 ELSE 0 END) AS product_b
    FROM Orders
    GROUP BY customer_id) AS t JOIN Customers
ON t.customer_id = Customers.customer_id
WHERE t.product_a>0 AND product_b >0 AND Customers.customer_id NOT IN (
    SELECT DISTINCT customer_id
    FROM Orders
    WHERE product_name = 'C')
ORDER BY Customers.customer_id;



# 1407. Top Travellers
# coalesce( window function ,0)
# 831 ms 
select distinct u.name,
coalesce(sum(r.distance) over(partition by r.user_id),0) as travelled_distance
from Users u
left join Rides r 
on r.user_id = u.id
order by travelled_distance desc, name asc;

# ifnull( window function ,0)
# 754 ms 
select distinct u.name,
ifnull(sum(r.distance) over(partition by r.user_id),0) as travelled_distance
from Users u
left join Rides r 
on r.user_id = u.id
order by travelled_distance desc, name asc;

# group by seems quicker 
# 673ms
select distinct u.name,
coalesce(sum(r.distance),0) as travelled_distance
from Users u
left join Rides r 
on r.user_id = u.id
group by r.user_id
order by travelled_distance desc, name asc;



# 1412. Find the Quiet Students in All Exams
# inner join Don’t return the student who has never taken any exam
# max() and min()
select distinct s.*
from Student s Inner join Exam e
on e.student_id = e.student_id
where s.student_id not in 
(
select distinct e.student_id
from (select student_id, exam_id, 
    min(score) over(partition by exam_id) as min_score,
    max(score) over(partition by exam_id) as max_score
    from Exam) e2
)
order by s.student_id;


#Solution 1:
WITH t1 AS(
SELECT student_id
FROM
(SELECT *,
MIN(score) OVER(PARTITION BY exam_id) AS least,
MAX(score) OVER(PARTITION BY exam_id) AS most
FROM exam) a
WHERE least = score OR most = score)

SELECT DISTINCT student_id, student_name
FROM exam JOIN student
USING (student_id)
WHERE student_id != all(SELECT student_id FROM t1)
order by 1



# 1421. NPV Queries
# take care null
# coalesce(n.npv, 0) as npv
# if(isnull(npv),0,npv)
select q.id,
q.year, 
coalesce(n.npv,0) as npv
from NPV n
right join Queries q 
on n.id = q.id and q.year = n.year



# 1435. Create a Session Bar Chart
#Solution 1:
(SELECT '[0-5>' AS bin,
 SUM(CASE WHEN duration/60 < 5 THEN 1 ELSE 0 END) AS total FROM Sessions)
 UNION
(SELECT '[5-10>' AS bin,
 SUM(CASE WHEN ((duration/60 >= 5) AND (duration/60 < 10)) THEN 1 ELSE 0 END) AS total FROM Sessions)
 UNION
(SELECT '[10-15>' AS bin,
 SUM(CASE WHEN ((duration/60 >= 10) AND (duration/60 < 15)) THEN 1 ELSE 0 END) AS total FROM Sessions)
 UNION
(SELECT '15 or more' AS bin,
 SUM(CASE WHEN duration/60 >= 15 THEN 1 ELSE 0 END) AS total FROM Sessions)

#Solution 2:
# union and where
SELECT '[0-5>' AS bin, count(1) AS total
FROM Sessions
WHERE duration>=0 AND duration < 300
UNION
SELECT '[5-10>' AS bin, count(1) AS total
FROM Sessions
WHERE duration>=300 AND duration < 600
UNION
SELECT '[10-15>' AS bin, count(1) AS total
FROM Sessions
WHERE duration>=600 AND duration < 900
UNION
SELECT '15 or more' AS bin, count(1) AS total
FROM Sessions
WHERE duration >= 900



# 1440. Evaluate Boolean Expression
# Solution 1:
# nested INNER JOIN can trim the volume of the intermediate table, 
# which gives us better performance
SELECT t.left_operand, t.operator, t.right_operand,
    (CASE WHEN v1_value>v2.value AND operator = '>' THEN "true"
          WHEN v1_value<v2.value AND operator = '<' THEN "true"
          WHEN v1_value=v2.value AND operator = '=' THEN "true"
          ELSE "false"
          END) AS value
FROM
   (SELECT e.*, v1.value AS v1_value
    FROM Expressions AS e INNER JOIN Variables AS v1
    ON e.left_operand = v1.name) AS t INNER JOIN Variables AS v2
    ON t.right_operand = v2.name;
    
#Solution 2:
SELECT t.left_operand, t.operator, t.right_operand,
    (CASE WHEN operator = '>' THEN IF(v1_value>v2.value, "true", "false")
          WHEN operator = '<' THEN IF(v1_value<v2.value, "true", "false")
          WHEN operator = '=' THEN IF(v1_value=v2.value, "true", "false")
          END) AS value
FROM
   (SELECT e.*, v1.value AS v1_value
    FROM Expressions AS e INNER JOIN Variables AS v1
    ON e.left_operand = v1.name) AS t INNER JOIN Variables AS v2
    ON t.right_operand = v2.name;    



# 1445. Apples & Oranges 
select sale_date,
sum(case when fruit = "apples" then +sold_num else -sold_num end) over(partition by sale_date) as diff
from Sales
order by sale_date;



# 1454. Active Users 
with t1 as 
(
	select id, login_date
    lead(login_date, 4) over(partition by id order by login_date) date_5
    from (select distinct * from logins) b
)

select distinct a.id, a.name 
from t1
inner join accounts a
on t1.id = a.id
where datediff(t1.date_5,login_date) = 4
order by id



# 1459. Rectangles Area
select t1.id as p1, 
	t2.id as p2,
    ABS(t1.x_value-t2.x_value)*ABS(t1.y_value-t2.y_value) as area
from Points as t1 
inner join Points as t2
on t1.id < t2.id and t1.x_value != t2.x_value and t1.y_value != t2.y_value
order by area desc, p1, p2;



# 1468. Calculate Salaries
# round( case when then else end , 0 )
# or case when then round(,0) when then round(,0)
with cte as( 
select company_id,
	max(salary) over(partition by company_id) as max_salary 
from Salaries)

select company_id, 
	employee_id,
	employee_name, 
	case when max_salary < 1000, then salary
	when max_salary >=1000 and max_salary <=10000 then round(salary*(1-0.24),0)
    when max_salary > 10000 then round(salary*(1-0.49),0) end as salary
from cte;

#Solution 1:
WITH t1 AS (
SELECT company_id, employee_id, employee_name, salary AS sa, MAX(salary) OVER(PARTITION BY company_id) AS maximum
FROM salaries)

SELECT company_id, employee_id, employee_name,
CASE WHEN t1.maximum<1000 THEN t1.sa
WHEN t1.maximum BETWEEN 1000 AND 10000 THEN ROUND(t1.sa*.76,0)
ELSE ROUND(t1.sa*.51,0)
END AS salary
FROM t1

#Soltion 2:
SELECT Salaries.company_id, Salaries.employee_id, Salaries.employee_name,
    ROUND(CASE WHEN salary_max<1000 THEN Salaries.salary
               WHEN salary_max>=1000 AND salary_max<=10000 THEN Salaries.salary * 0.76
               ELSE Salaries.salary * 0.51 END, 0) AS salary
FROM Salaries INNER JOIN (
    SELECT company_id, MAX(salary) AS salary_max
    FROM Salaries
    GROUP BY company_id) AS t
ON Salaries.company_id = t.company_id



# 1479. Sales by Day of the Week
# dayname() = 'Monday'
# using (item_id)
with t1 as
(
	select distinct item_category,
    case when dayname(order_date)='Monday' then sum(quantity) over(partition by item_category, dayname(orderdate)) else 0 end as Monday,
    case when dayname(order_date)='Tuesday' then sum(quantity) over(partition by item_category, dayname(orderdate)) else 0 end as Tuesday,
    case when dayname(order_date)='Wednesday' then sum(quantity) over(partition by item_category, dayname(orderdate)) else 0 end as Wednesday,
    case when dayname(order_date)='Thursday' then sum(quantity) over(partition by item_category, dayname(orderdate)) else 0 end as Thursday,
    case when dayname(order_date)='Friday' then sum(quantity) over(partition by item_category, dayname(orderdate)) else 0 end as Friday,
    case when dayname(order_date)='Saturday' then sum(quantity) over(partition by item_category, dayname(orderdate)) else 0 end as Saturday,
    case when dayname(order_date)='Sunday' then sum(quantity) over(partition by item_category, dayname(orderdate)) else 0 end as Sunday
	from Orders o 
    right join item i
    using (item_id)
) 

select item_category as Category, 
Monday, 
Tuesday, 
Wednesday, 
Thursday, 
Friday, 
Saturday, 
Sunday
from t1
order by category；



# 1484. Group Sold Products By The Date
# the trick part is group_concat(distinct  ) as three rows => one rows
# you can't use count(distinct ) with window function like
# this question they specify ‘There is no primary key for this table’
# so we have to use distinct here  
select distinct sell_date,
count(distinct product) as num_sold,
group_concat(distinct product) as products
from Activities
group by sell_date
order by sell_date;



# 1495. Friendly Movies Streamed Last Month
select distinct c.title
from TVProgram t
inner join Content c
on c.content_id = t.content_id  
where month(t.program_date) =6 and year(t.program_date) = 2020 
and c.Kids_content = "Y" 
and c.content_type = 'Movies';



# 1501. Countries You Can Safely Invest In
# substring(phone_number from 1 for 3)
with t1 as
(
	select caller_id as id, duration as total
    from
    (
		select caller_id, duration
        from calls
        union all
        select callee_id, duration
        from calls
    ) a
)

select name as country
from
(select distinct avg(total) over(partition by code) as avg_call, 
	avg(total) over() as global_avg,
    c.name
from 
((select *, 
coalesce(total,0) as duration,
substring(phone_number from 1 for 3) as code
from person 
right join t1
using (id)) b
join country c
on c.country_code = b.code)) d
where avg_call > global_avg



# 1511. Customer Order Frequency 
# Solution 1:
# having ( sum()>=10 and sum()>=10 )
# sum(case when like '2020-06%' then else end )
SELECT o.customer_id, c.name
FROM Customers c, Product p, Orders o
WHERE c.customer_id = o.customer_id AND p.product_id = o.product_id
GROUP BY o.customer_id
HAVING
(
    SUM(CASE WHEN o.order_date LIKE '2020-06%' THEN o.quantity*p.price ELSE 0 END) >= 100
    and
    SUM(CASE WHEN o.order_date LIKE '2020-07%' THEN o.quantity*p.price ELSE 0 END) >= 100
);

#Solution 3:
SELECT o.customer_id, name
JOIN Product p
ON o.product_id = p.product_id
JOIN Customers c
ON o.customer_id = c.customer_id
GROUP BY 1, 2
HAVING SUM(CASE WHEN date_format(order_date, '%Y-%m')='2020-06'
THEN price*quantity END) >= 100
AND
SUM(CASE WHEN date_format(order_date, '%Y-%m')='2020-07'
THEN price*quantity END) >= 100;

#Solution 2:
SELECT customer_id, name
FROM
(
    SELECT o.customer_id, c.name,
        sum(CASE WHEN left(o.order_date,7) = '2020-06' THEN p.price * o.quantity END) AS JuneSpend,
        sum(CASE WHEN left(o.order_date,7) = '2020-07' THEN p.price * o.quantity END) AS JulySpend
    FROM Orders o
    LEFT JOIN Customers c ON o.customer_id = c.customer_id
    lEFT JOIN Product p ON o.product_id = p.product_id
    GROUP BY o.customer_id
    HAVING JuneSpend >= 100 AND JulySpend >= 100
) AS temp;




# 1517. Find Users With Valid E-Mails 
SELECT * FROM Users
WHERE regexp_like(mail, '^[A-Za-z]+[A-Za-z0-9\_\.\-]*@leetcode.com');



# 1527. Patients With a Condition
# the trick part is where conditions like 'DIAB1%' or conditions like '% DIAB1%' 
# weird question...
select patient_id,
patient_name,
conditions
from Patients
where conditions like 'DIAB1%' or conditions like '% DIAB1%';



# 1532. The Most Recent Three Orders
# where r_num <= 3
select 
customer_id, 
customer_name, 
order_id, 
order_date
from (
		select 
		c.customer_id, 
		c.name as customer_name, 
		o.order_id, 
		o.order_date,
		row_number() over(partition by o.customer_id, order by o.order_date desc) as r_num
		from Customers c
		join Orders o
		on c.customer_id = o.customer_id
) temp
where r_num <= 3
order by customer_name asc, customer_id asc, order_date desc;



# 1543. Fix Product Name Format
# trim() remove white space
# lower() lowercase
# date_format()
select trim(lower(product_name)) as product_name,
date_format(sale_date,'YYYY-MM') as sale_date,
count(sale_id) over(partition by trim(lower(product_name)), date_format(sale_date,'YYYY-MM')) as total
from Sales
order by trim(lower(product_name)) asc, date_format(sale_date,'YYYY-MM')  asc;

SELECT TRIM(LOWER(product_name)) AS product_name,
       DATE_FORMAT(sale_date, '%Y-%m') AS sale_date,
       COUNT(*) AS total
FROM Sales
GROUP BY 1, DATE_FORMAT(sale_date, '%Y-%m')
ORDER BY 1, 2;



# 1549. The Most Recent Orders for Each Product
select product_name, 
	product_id, 
	order_id, 
	order_date,
from 
(
	select p.product_name, 
	p.product_id, 
	o.order_id, 
	o.order_date,
	rank() over(partition by product_id, order_date order by order_date desc) as rnk
	from Products p
	join Orders o 
	on o.product_id = p.product_id
) temp
where rnk = 1
order by product_name, product_id, order_id;



# 1555. Bank Account Summary
# +t.amount as trans
# ifnull(,0)
#
select user_id, 
user_name,
credit + ifnull(sum(trans) over(partition by user_id),0) as credit,
case when credit + ifnull(sum(trans) over(partition by user_id),0)<0 then "Yes" else "No" end as credit_limit_breached
from(
	select u.user_id, u.user_name, u.credit, +t.amount as trans
	from Users u
	join Transaction t on u.user_id = t.paid_to
	union all
	select u.user_id, u.user_name, u.credit, -t.amount as trans
	from Users u
	join Transaction t on u.user_id = t.paid_by
) temp;



# 1565. Unique Orders and Customers Per Month
select date_format(order_date,'%Y-%m') as month, 
count(order_id) over(partition by year(order_date), month(order_date)) as order_count, 
count(distinct customer_id) over(partition by year(order_date), month(order_date)) as as customer_count
from Orders
where invoices > 20;

# Solution 1:
SELECT LEFT(order_date, 7) AS month, COUNT(DISTINCT order_id) AS order_count,
	COUNT(DISTINCT customer_id) AS customer_count
FROM orders
WHERE invoice > 20
GROUP BY month;



# 1571. Warehouse Manager
select w.name as warehouse_name,
	sum(p.Width*p.Length*p.Height*w.units) over(partition by w.name) as volume
from Warehouse w
left join Products p
on w.product_id = p.product_id;



# 1581. Customer Who Visited but Did Not Make Any Transactions
# distinct + window function = group by
# don't forget include the null in where clause
select distinct v.customer_id,
count(v.visit_id) over(partition by v.customer_id) as count_no_trans
from Visits v
left join Transactions t
on v.visit_id = t.visit_id
where transaction_id is null and amount is null;

select customer_id as customer_id,
count(v.visit_id) as count_no_trans
from Visits v
left join Transactions t
on v.visit_id = t.visit_id
where transaction_id is null and amount is null
group by customer_id



# 1587. Bank Account Summary II
# the trick part is you can't use window function with having
# sum() over() as balance having balance is not allow
# sum() as balance group by having balance is Okay 
# also you can't do order before having
# plus you don't need distinct if it's primary key
select u.name,
sum(t.amount)  as balance
from Users u
join Transactions t
on t.account = u.account
group by t.account 
having balance > 10000; -- use having sum(t.amount) > 10000 is okay too!

#Solution 2:
WITH tmp AS(
SELECT t.account, u.name, SUM(amount) AS balance
FROM Transactions t
LEFT JOIN Users u ON t.account = u.account
GROUP BY account )

SELECT name, balance
FROM tmp
WHERE balance > 10000；



# 1596. The Most Frequently Ordered Products for Each Customer
select customer_id, 
product_id, 
product_name
from (
	select t1.*,
		p.product_id, 
		p.product_name,
		rank() over(partition by customer_id, cnt order by cnt desc) as rnk
	from (
		select *,
		count(order_id) over (partition by customer_id, product_id order by product_id) as cnt
		from Orders 
	) t1
	join Products p
	on p.product_id = t1.product_id
) t2
where rnk = 1;

# solution- 2:
#  RANK() OVER( PARTITION BY   ORDER BY COUNT(*) DESC )
SELECT customer_id, T.product_id, product_name
FROM(
    SELECT customer_id, product_id,
    RANK() OVER( PARTITION BY customer_id ORDER BY COUNT(*) DESC ) AS RK
    FROM Orders o
    GROUP BY customer_id, product_id
) T
LEFT JOIN Products p on p.product_id = t.product_id
WHERE RK=1；



# 1607. Sellers With No Sales
select seller_name
from Seller
where seller_id not in(
				select distinct seller_id 
                from Orders
                where year(sale_date) = '2020')
order by seller_name asc;



# 1613. Find the Missing IDs
# recursive
WITH RECURSIVE CTE AS(
    SELECT 1 AS 'id', MAX(c.customer_id) AS 'Max_Id'
    FROM Customers c
    UNION ALL
    SELECT id+1, Max_Id
    FROM CTE
    WHERE id < Max_id
)

SELECT id AS 'ids'
FROM CTE c
WHERE c.id NOT IN (SELECT customer_id FROM Customers)
ORDER BY 1 ASC;



# 1623. All Valid Triplets That Can Represent a Country
# <> !=
select a.student_name as 'member_A',
b.student_name as 'member_B',
c.student_name as 'member_C'
from SchoolA as a
join SchoolB as b
on a.student_id <> b.student_id and a.student_name <> b.student_name
join SchoolC as c
on a.student_id <> c.student_id and a.student_name <> c.student_name
and b.student_id <> c.student_id and b.student_name <> c.student_name;



# 1633. Percentage of Users Attended a Contest 
select round(count(r.user_id) over(partition by r.contest_id)/count(u.*)*100,2) as 
from Users as u, Register as r
order contest_id asc;



# 1635. Hopper Company Queries I 
# better option 
# window function with unbounded row
select month,
 sum(month_drivers*1.0) over(partition by month order by month rows between unbounded preceding and current row) as active_drivers,
coalesce(accepted_rides,0) as accepted_rides
from (
	select month(join_date) as month,
    count(d.driver_id) over(partition by month) as month_drivers
	from Drivers d 
	where year(join_date) <= 2020
) active_drivers
left join (
			select month(r.requested_at) as 'month', 
			count(ar.driver_id) over(partition by month(r.requested_at)) as 'accepted_rides'
			From Riders r
			join AcceptedRides ar
			on r.ride_id = ar.ride_id
			where year(requested_at) = '2020'
			) active_rides
on active_drivers.month = active_rides.month
order by month asc;

# (CASE WHEN year(d.join_date) = 2019 THEN '1' ELSE month(d.join_date) END)
# (SELECT 1 AS 'month')
# union
# where year(join_date) <= 2020
select month,
 count(d.driver_id) over(partition by month) as active_drivers,
coalesce(accepted_rides,0) as accepted_rides
from (
	select d.driver_id,
    (CASE WHEN year(d.join_date) = 2019 THEN '1' ELSE month(d.join_date) END) as 'month'
	from Drivers d 
	right join
	(
			(SELECT 1 AS 'month')
    UNION (SELECT 2 AS 'month')
    UNION (SELECT 3 AS 'month')
    UNION (SELECT 4 AS 'month')
    UNION (SELECT 5 AS 'month')
    UNION (SELECT 6 AS 'month')
    UNION (SELECT 7 AS 'month')
    UNION (SELECT 8 AS 'month')
    UNION (SELECT 9 AS 'month')
    UNION (SELECT 10 AS 'month')
    UNION (SELECT 11 AS 'month')
    UNION (SELECT 12 AS 'month')
	) as months
    on months.month <= d.month
	where year(join_date) <= 2020
) active_drivers
left join (
			select month(r.requested_at) as 'month', 
			count(ar.driver_id) over(partition by month(r.requested_at)) as 'accepted_rides'
			From Riders r
			join AcceptedRides ar
			on r.ride_id = ar.ride_id
			where year(requested_at) = '2020'
			) active_rides
on active_drivers.month = active_rides.month
order by month asc;



# 1645. Hopper Company Queries II 
# better option 
# window function with unbounded row
select month,
round(coalesce(100*coalesce(accepted_rides,0)/active_drivers,0), 2) as working_percentage
from
(
	select month, 
    sum(month_drivers*1.0) over(partition by month order by month rows between unbounded preceding and current row) as active_drivers
	from 
    (
		select month(join_date) as month,
		count(d.driver_id) over(partition by month(join_date)) as month_drivers
		from Drivers d 
		where year(join_date) <= 2020
	) 
) active_drivers
left join (
			select month(r.requested_at) as month, 
			count(ar.driver_id) over(partition by month(r.requested_at)) as accepted_rides
			From Riders r
			join AcceptedRides ar
			on r.ride_id = ar.ride_id
			where year(requested_at) = '2020'
			) active_rides
active_drivers.month = active_rides.month
order by month;

# ifnull(,0) null -> 0
# coalesce(,0)  0 -> 0
# dealing with cumulative union
# on d.join_date <= m.day 
select month,
round(coalesce(100*coalesce(accepted_rides,0)/active_drivers,0), 2) as working_percentage
from
(
	select month, count(d.driver_id) over(partition by month(d.join_date)) as active_drivers
	from Drivers d 
	right join
	(
			SELECT "2020-1-31" AS day, 1 AS month
			UNION SELECT "2020-2-29", 2 AS month
			UNION SELECT "2020-3-31", 3 AS month
			UNION SELECT "2020-4-30", 4 AS month
			UNION SELECT "2020-5-31", 5 AS month
			UNION SELECT "2020-6-30", 6 AS month
			UNION SELECT "2020-7-31", 7 AS month
			UNION SELECT "2020-8-31", 8 AS month
			UNION SELECT "2020-9-30", 9 AS month
			UNION SELECT "2020-10-31", 10 AS month
			UNION SELECT "2020-11-30", 11 AS month
			UNION SELECT "2020-12-31", 12 AS month
	) as months
	on d.join_date <= m.day
) active_drivers
left join (
			select month(r.requested_at) as month, 
			count(ar.driver_id) over(partition by month(r.requested_at)) as accepted_rides
			From Riders r
			join AcceptedRides ar
			on r.ride_id = ar.ride_id
			where year(requested_at) = '2020'
			) active_rides
active_drivers.month = active_rides.month
order by month;



# 1651. Hopper Company Queries III
# avg() partition by month order by month rows between current row and 2 following
select month,
 coalesce(round(avg(month_dis*1.0) over (partition by month order by month rows between current row and 2 following),2),0)  as 'average_ride_distance',
 coalesce(round(avg(month_dur*1.0) over (partition by month order by month rows between current row and 2 following),2),0)  as 'average_ride_duration'
from
(
	select month(r.requested_at) as 'month', 
    sum(ar.ride_distance) over(partition by month(r.requested_at) order by month(r.requested_at)) as month_dis
    sum(ar.ride_duration) over(partition by month(r.requested_at) order by month(r.requested_at)) as month_dur
	from Rides r
    left join AcceptedRides ar
    on r.ride_id = ar.ride_id
    where year(r.requsted_at) = '2020'
) temp1
order by month asc;

# solution
SELECT month,
    COALESCE(ROUND(SUM(ride_distance)/3,2),0) AS average_ride_distance,
    COALESCE(ROUND(SUM(ride_duration)/3,2),0) AS average_ride_duration
FROM
(
    SELECT months.month, ride_id
    FROM Rides
    RIGHT JOIN
    (
        SELECT "2020-1-1" AS start, "2020-3-31" AS last, 1 AS month
        UNION SELECT "2020-2-1", "2020-4-30", 2
        UNION SELECT "2020-3-1", "2020-5-31", 3
        UNION SELECT "2020-4-1", "2020-6-30", 4
        UNION SELECT "2020-5-1", "2020-7-31", 5
        UNION SELECT "2020-6-1", "2020-8-31", 6
        UNION SELECT "2020-7-1", "2020-9-30", 7
        UNION SELECT "2020-8-1", "2020-10-31", 8
        UNION SELECT "2020-9-1", "2020-11-30", 9
        UNION SELECT "2020-10-1", "2020-12-31", 10
    ) AS months
    ON months.start <= requested_at AND months.last >= requested_at
) total
LEFT JOIN AcceptedRides AS a
ON total.ride_id=a.ride_id
GROUP BY month
ORDER BY month;



# 1661. Average Time of Process per Machine
select machine_id, 
round(avg(process_time) over(partition by machine_id),3) as processing_time
from (
select machine_id,
process_id,
sum(case when activity_type = 'start' then timestamp else -timesamp) over(partition by machine_id, process_id order by machine_id, process_id) as process_time,
from Activity
) temp
order by machine_id;

# better option
select machine_id,
	round(sum(if(activity_type='start',timestamp,-timestamp)) over(partition by machine_id)/count(distinct process_id) over(partition by machine_id),3) as processing_time
from Activity
order by machine_id;



# 1667. Fix Names in a Table
# the trick part is substring(name,1,1) and substring(name,2)
# then upper() and lower() 
# then concat();
select user_id,
concat(upper(substring(name,1,1)),lower(substring(name,2))) as name
from Users
order by user_id;



# 1677. Product’s Worth Over Invoices 
# sum not count
select p.name,
sum(i.rest) over(partition by i.product_id) rest,
sum(i.paid) over(partition by i.product_id) paid,
sum(i.canceled) over(partition by i.product_id) canceled,
sum(i.refunded) over(partition by i.product_id) refunded
from Invoice i
join Product p
on i.product_id = p.procduct_id
order by product_name;



# 1683. Invalid Tweets
select tweet_id
from Tweets
where length(content) > 15;



# 1693. Daily Leads and Partners
# the trick part is you can't use distinct with window functions 
# count(distinct ) over(partition by) is not allow
# count(distinct ) group by is okay
select distinct date_id,
make_name,
count(distinct lead_id) as unique_leads,
count(distinct partner_id)as unique_partners
from DailySales
group by date_id, make_name;


# 1699. Number of Calls Between Two Persons
select distinct person1, person2,
count(duration) over(partition by person1,person2) as call_count,
sum(duration) over(partition by person1,person2) as total_duration
from (
	select from_id as person1,
	to_id as person2,
	duration
	from Calls
	where from_id < to_id
	union 
	select to_id as person1,
	from_id as person2,
	duration
	from Calls
	where to_id < from_id
) temp;

# better option 
select if(from_id<to_id, from_id, to_id) as person1,
	if(from_id>to_id, from_id, to_id) as person2,
    count(duration) as call_count,
    sum(duration) as total_duration
from Calls
group by if(from_id<to_id,from_id,to_id),
    if(from_id>to_id,from_id,to_id);



# 1709. Biggest Window Between Visits
select user_id,
max(datediff(visit_date,next_visit)) over(partition by user_id) as biggest_window
from 
(
	select *,
	coalesce(lead(visit_date,1)over(order by user_id),'2021-01-01') as next_visit
	from UserVisits
) temp
order by user_id;

# coalesce(lead(visit_date,1)over(order by user_id),'2021-01-01') 
select user_id, max(diff) over(partition by user_id) as biggest_window
from
(
	select user_id,
    datediff(coalesce(lead(visit_date) over(partition by user_id order by visit_date),'2021-01-01'),visit_date) as diff
) t
order by user_id;



# 1715. Count Apples and Oranges 
select 
sum(if(b.chest_id = null, b.apple_count,b.apple_count+c.apple_count)) as apple_count,
sum(if(b.chest_id = null, b.orange_count,b.orange_count+c.orange_count)) as orange_count
from Boxes b
left join Chests c 
on b.chest_id = c.chest_id;



# 1729. Find Followers Count
# distinct + partition by = group
select distinct user_id,
count(follower_id) over(partition by user_id) as followers_count
from Followers
order by user_id;

select user_id,
count(follower_id) as followers_count
from Followers
group by user_id
order by user_id;



# 1731. The Number of Employees Which Report to Each Employee
select employee_id,
name,
count(employee_id) over(partition by reports_to) as reports_count,
round(avg(age) over(partition by reports_to),0) as average_age
from Employees
where employee_id in (select distinct reports_to from Employees)
order by employee_id;

select e1.reports_to as employee_id,
e2.name,
count(e1.reports_to) as reports_count,
round(avg(e1.age),0) as average_age
from employees e1
join employees e2
on e1.reports_to = e2.employee_id
group by e1.reports_to
order by e1.reports_to;



# 1741. Find Total Time Spent by Each Employee
# the trick part is distinct
select distinct event_day as day,
emp_id,
sum(out_time-in_time) over(partition by emp_id, event_day) as total_time
from Employees;

# distinct + partition by = group by
select event_day as day,
emp_id,
sum(out_time-in_time) as total_time
from Employees
group by emp_id, event_day;



# 1747. Leetflex Banned Accounts
select distinct l1.account_id
from LogInfo l1
Join LogInfo l2
on l1.account_id = l2.account_id and l1.ip_address != l2.ip_address
where not(l1.login > l2.logout or l1.logout < l2.login);



# 1757. Recyclable and Low Fat Products 
select product_id
from Products
where low_fats = 'Y' and recyclable = 'Y';



# 1767. Find the Subtasks That Did Not Execute 
with recursive cte as (
	select 1 as subtask_id
	union all 
	select subtask_id + 1
	from cte
	where subtask_id < (select max(subtasks_count) from Tasks)
)
select t.task_id, c.subtask_id
from cte as c
inner join Tasks as t 
on cte.subtask_id <= tasks.subtasks_count
left join Executed as e 
on t.task_id = e.task_id and c.subtask_id = e.subtask_id
where e.subtask_id is null;



# 1777. Product’s Price for Each Store
select product_id, 
case when store = 'store1' then price else null end as store1,
case when store = 'store2' then price else null end as store2,
case when store = 'store3' then price else null end as store3
from Products;



# 1783. Grand Slam Titles
# sum(if( = , 1, 0) + ..)
select p.player_id,
p.player_name,
sum(if(Wimbledon = player_id, 1, 0)+
	if(Fr_open = player_id, 1, 0)+ 
    if(US_open = player_id, 1, 0)+
    if(Au_open = player_id, 1, 0)) as grand_slams_count
from Players p
inner join Championships c
on p.player_id = c.Wimledon 
or p.player_id = c.Fr_open
or p.player_id = c.US_open
or p.player_id = c.Au_open;



# 1789. Primary Department for Each Employee
select employee_id, 
department_id
from Employee
where primary_flag = 'Y' or employee_id in 
				(
				select employee_id
                from employee
                group by employee_id
                having count(department_id) = 1
				);



# 1795. Rearrange Products Table
# this question the trick part is union and is not null
select product_id, 
case when store1 then 'store1' end as 'store',
case when store1 then store1 end as 'price'
from Products
where store1 is not null
union 
select product_id, 
case when store2 then 'store2' end as 'store',
case when store2 then store2 end as 'price'
from Products
where store2 is not null
union
select product_id, 
case when store3 then 'store3' end as 'store',
case when store3 then store3 end as 'price'
from Products
where store3 is not null;



# 1809. Ad-Free Sessions

select distinct session_id
from Playback p
full outer join Ads a 
on a.timestamp between p.end_time and p.start_time and a.customer_id = p.customer_id
where a.ad_id is null;

select session_id
from Playback p
left join Ads a 
on a.timestamp between p.end_time and p.start_time and a.customer_id = p.customer_id
group by session_id
having count(a.ad_id) = 0;



# 1811. Find Interview Candidates
with t2 as(
select user_id
from (
		select u.user_id, 
			c.contest_id,
			case when gold_medal = user_id then 1 
			case when silver_medal = user_id then 1
			case when bronze_medal = user_id then 1
			else 0 end as metalcount, 
            lead(contest_id,2) OVER(PARTITION BY user_id ORDER BY contest_id) as cont_3,
            count(contest_id) over (partition by gold_medal) as golds
		from Users u
		left join Contests c 
		on u.user_id in (c.gold_medal, c.silver_medal, c.bronze_medal) 
        ) as t
where metalcount = 1 
and cont_3 - contest_id = 3

union 

select distinct gold_medal as user_id
from t 
where golds >=3
)

select u.name, u.mail
from Users u
join t2
where t2.user_id = u.user_id;

# other solution
# join using( );
with t as 
(
    SELECT *,  
        contest_id -
        RANK() over (PARTITION BY user_id, user_id in (gold_medal, silver_medal, bronze_medal) order by contest_id) consecutive_ref
    FROM Contests JOIN Users on user_id in (gold_medal, silver_medal, bronze_medal)
    order by user_id, contest_id
),
u as 
(
    SELECT user_id 
    from t 
    GROUP BY 1
    having sum(user_id=gold_medal)>=3
        union 
    select user_id
    FROM t 
    GROUP BY user_id,consecutive_ref
    HAVING count(1) >=3
)
SELECT name, mail FROM u LEFT JOIN Users using(user_id);



# 1821. Find Customers With Positive Revenue this Year
select customer_id
from Customers
where revenue > 0 and year = 2021;



# 1831. Maximum Transaction Each Day
select transaction_id 
from(
	select *,
    max(amount) over(partition by date(day)) as max_amount
    from Transactions t
) as t1
where amount = max_amount
order by transaction_id;

select transaction_id 
from(
	select *,
    rank() over(partition by date(day) order by amount desc) as rnk
    from Transactions t
) as t1
where rnk = 1
order by transaction_id;



# 1841. League Statistics
# sum(if( = , 1, 0) + ..)
with t2 as (
	select t.*,
		m.*,
		case when home_team_goals = away_team_goals then 1
		case when home_team_goals < away_team_goals then 3
		case when home_team_goals > away_team_goals then 0 end as away_team_points,
		
		case when home_team_goals = away_team_goals then 1
		case when home_team_goals < away_team_goals then 0
		case when home_team_goals > away_team_goals then 3 end as home_team_points,
		
		sum( if(home_team_id = team_id, home_team_goals , 0) +  
				if(away_team_id = team_id, away_team_goals, 0) ) as goal_for,
				
		sum( if(home_team_id = team_id, away_team_goals , 0) +  
				if(away_team_id = team_id, home_team_goals, 0) ) as goal_against,
		
		sum( if(home_team_id = team_id, 1 , 0) +  
				if(away_team_id = team_id, 1 , 0) ) as matches_played

	from Matches m
	join Teams t on t.team_id = m.home_team_id
)

select distinct team_name,
	matches_played,
		sum( if(home_team_id = team_id, home_team_points , 0) +  
			if(away_team_id = team_id, away_team_points , 0) ) as points,
            goal_for,
			goal_against,
            goal_for - goal_against as goal_diff
from t2
order by points desc, goal_diff desc, team_name;

# solution 2 better one
# sum( case when then if( = , , ))
# sum( if (t.team_id = x, x - y, y - x) )
# sum( if(t.team_id = m.home_team_id, home_team_goals - away_team_goals, away_team_goals - home_team_goals) )
# join on = or =
SELECT team_name
    ,count(1) matches_played
    ,sum(case when home_team_goals = away_team_goals then 1 
            when home_team_goals > away_team_goals then if(t.team_id = m.home_team_id,3,0)
            when home_team_goals < away_team_goals then if(t.team_id = m.home_team_id,0,3)
            else null end) points
    ,sum(if(t.team_id = m.home_team_id,home_team_goals,away_team_goals)) goal_for
    ,sum(if(t.team_id = m.home_team_id,away_team_goals,home_team_goals)) goal_against
    ,sum(if(t.team_id = m.home_team_id,home_team_goals-away_team_goals,away_team_goals-home_team_goals)) goal_diff
FROM Teams t INNER JOIN Matches m on t.team_id = m.home_team_id or t.team_id = m.away_team_id
GROUP BY team_name
ORDER BY points desc, goal_diff desc, team_name;



# 1843. Suspicious Bank Accounts
# self join
with temp as(
	select 
		a.account_id, 
		month(day) as month,
		sum(if(type = 'Creditor', amount, 0)) over(partition by account_id, month(day) order by account_id, month(day) asc) - max_income as income_diff
	from Transactions t
    join Accounts a
    on a.account_id = t.account_id
)
select distinct account_id
from temp t2 join temp t3 on t2.account_id = t3.account_id  
where income_diff > 0 and period_diff(t3.month, t2.month) = 1 
order by transaction_id asc;

# lead window function might not working only with month bc what if 12 -> 1
with t1 as(
	select 
		a.account_id, 
		month(day) as month,
		sum(if(type = 'Creditor', amount, 0)) over(partition by account_id, month(day) order by account_id, month(day) asc) - max_income as income_diff
	from Transactions t
    join Accounts a
    on a.account_id = t.account_id
),
t2 as
(
	select
	account_id, month(day) as month,
    lead(month(day),1) OVER(PARTITION BY account_id ORDER BY month(day)) as cont_2
    from t1
    where income_diff > 0
)

select distinct account_id
from t2
where cont_2 - month = 1
order by transaction_id asc;

# other solution
# sum(amount) > max(max_income) as exceed
# period_diff(m1, m2) = 1
# a1.exceed + a2.exceed =2
with a as (
    SELECT account_id, 
		DATE_FORMAT(day,'%Y%m') mon,
        sum(amount) > max(max_income)  exceed 
    from Transactions LEFT JOIN Accounts USING(account_id)
    where type='Creditor'
    GROUP BY account_id, DATE_FORMAT(day,'%Y%m')
    order by 1,2
)
SELECT DISTINCT a1.account_id 
FROM a a1, a a2 
where a1.account_id = a2.account_id and PERIOD_DIFF(a1.mon,a2.mon)=1 and a1.exceed + a2.exceed =2



# 1853. Convert Date Format
select date_format(day, '%W, %M %e, %Y') as day
from Days



# 1867. Orders With Maximum Quantity
# this is a mistake I made, I calculated
# the average quantity of this order. 
# so I add max(average)
select order_id
from(
	select *, 
	sum(quantity) / count(product_id) as average, 
	max(quantity) as maximum
	from OrdersDetails o
	group by order_id
	order by order_id
 ) as temp
 where maximum > max(average);
 
# option 2
select order_id
from OrdersDetails
group by order_id
having max(quantity) > (select  
max(sum(quantity) / count(product_id)) as max_average from OrdersDetails group by order_id)
order by order_id;

 
 # other options
 # An imbalanced order is one whose maximum quantity is strictly greater than 
 # the average quantity of every order (including itself).
 # 						   ^^^^^ ---> tricky part
 # every is sort of like any
 # having max() > all()
 select order_id from OrdersDetails
    group by order_id
    having max(quantity) > all(
        select sum(quantity) / count(distinct product_id)
        from OrdersDetails
        group by order_id
    );
    
# other options
# I guess use avg(quantity) would works too?
with t AS
(
    SELECT order_id, MAX(quantity) quan_max, MAX(AVG(quantity)) OVER() max_of_all_avg
    FROM OrdersDetails
    GROUP BY order_id
)
SELECT order_id
FROM t WHERE quan_max>max_of_all_avg



# 1873. Calculate Special Bonus
# the trick part is using case when instead of where
# because the output include the employee_id that with 0 bonus
select distinct employee_id,
case when employee_id%2=1 and substring(name,1,1)!='M' then salary else 0 end as bonus
from Employees
order by employee_id;



# 1875. Group Employees of the Same

select employee_id,
	name,
    salary,
    dense_rank() OVER(ORDER BY salary ASC) as team_id -- don't use parition by if unnecessary
from(
select *,
	count(employee_id) over(order by salary asc) as salary_count -- don't use parition by if unnecessary
from Employees
) as temp
where salary_count >= 2
order by team_id, employee_id asc;

# other options
# where exists()
select employee_id, name, salary, dense_rank() over (order by salary asc) as team_id
    from Employees as e1
    where exists (
        select * from Employees as e2
            where e1.employee_id != e2.employee_id
            and e1.salary = e2.salary
    )
    order by team_id, employee_id;

# other options
# row_number() I guess would work too
with t as (
    SELECT salary, ROW_NUMBER() OVER(ORDER BY salary) team_id 
    FROM employees
    GROUP BY salary
    HAVING COUNT(1) >1 
)
SELECT e.*, t.team_id
FROM t LEFT JOIN Employees e ON t.salary = e.salary
ORDER BY team_id, employee_id;



# 1890. The Latest Login in 2020
# the trick part is the latest login so using the max()
select distinct user_id, 
    max(time_stamp) over(partition by user_id) as last_stamp
from Logins
where year(time_stamp) = '2020';

select user_id,
max(time_stamp) as last_stamp
from Logins
where year(time_stamp) = '2020'
group by user_id; -- group by after where



# 1892. Page Recommendations II
# works but possible TLE
with user_friend as 
(
    SELECT user1_id, user2_id FROM Friendship
    union 
    SELECT user2_id, user1_id FROM Friendship
)
    SELECT f.user1_id as user_id, l.page_id, count(DISTINCT f.user2_id) friends_likes
    FROM user_friend f 
    LEFT JOIN Likes l on f.user2_id = l.user_id 
    where (user1_id, page_id) not in (
										SELECT * 
                                        from Likes 
                                        where user_id = f.user1_id and page_id = l.page_id
                                        )
    GROUP BY f.user1_id, l.page_id;

# other options 
# union here is so important for this 'friends' question
select f.user1_id as user_id, l.page_id, count(*) as friends_likes
    from (
        select user1_id, user2_id from Friendship union select user2_id, user1_id from Friendship
    ) as f
    inner join Likes l
    on f.user2_id = l.user_id 
    where not exists (
        select * from Likes where user_id = f.user1_id and page_id = l.page_id
    )
    group by f.user1_id, l.page_id;

# other options 
# left join is faster!
with user_friend as 
(
    SELECT user1_id, user2_id FROM Friendship
    union 
    SELECT user2_id, user1_id FROM Friendship
)
SELECT f.user1_id user_id, l2.page_id page_id, count(DISTINCT f.user2_id) friends_likes
FROM user_friend f 
LEFT JOIN Likes l2 on f.user2_id = l2.user_id  -- friends like
LEFT JOIN Likes l1 on f.user1_id = l1.user_id and l1.page_id = l2.page_id  -- we all like
where l1.page_id is null -- remove the user_id likes
GROUP BY f.user1_id, l2.page_id
order by f.user1_id



# 1907. Count Salary Categories
select distinct category, 
	ifnull(count(account_id) over(partition by category),0) as accounts_count
from (
	select *, case when income < 20000 then 'Low Salary'
		case when income > 50000 then 'High Salary'
		else 'Average Salary' end as category
	from Accounts
) as temp;

# other options
SELECT 'Low Salary' category,sum(if(income<20000,1,0)) accounts_count FROM accounts
UNION 
SELECT 'Average Salary' category,sum(if(income between 20000 and 50000,1,0)) accounts_count FROM accounts
UNION 
SELECT 'High Salary' category,sum(if(income>50000,1,0)) accounts_count FROM accounts;



# 1917. Leetcodify Friends Recommendations
select user_id, 
	recommended_id
from (
	select distinct l1.user_id, l2.user_id as recommended_id, count(distinct l1.song_id) over(partition by l1.user_id, l1.day) as cnt
	from Listens l1
	inner join Listens l2
	on l1.day = l2.day and l1.song_id = l2.song_id and l1.user_id <> l2.user_id
	where (l1.user_id,  l2.user_id) not in (	select user1_id, user2_id
												from Friendship 
												union
												select user2_id, user1_id
												from Friendship
											)
) as temp
where cnt >= 3

# modify the above to this short version
select distinct l1.user_id, l2.user_id as recommended_id
from Listens l1
inner join Listens l2
on l1.day = l2.day and l1.song_id = l2.song_id and l1.user_id <> l2.user_id
where (l1.user_id,  l2.user_id) not in (	select user1_id, user2_id
										from Friendship 
										union
										select user2_id, user1_id
										from Friendship
									)
group by l1.user_id, l1.day
having count(distinct l1.song_id) >= 3

# other options
with rec as (
SELECT user_id, recommended_id
FROM
(
	SELECT l1.user_id user_id, l2.user_id recommended_id, l2.song_id, l2.day
	FROM  Listens l1 INNER JOIN Listens l2 on 
	l1.user_id != l2.user_id AND 
	l1.song_id = l2.song_id AND 
	l1.`day`=l2.`day`
	ORDER BY user_id, recommended_id
) t
GROUP BY user_id, recommended_id, day
HAVING COUNT(DISTINCT song_id) >=3 
)

SELECT * 
from rec r1	
WHERE (user_id, recommended_id) not in (SELECT user1_id, user2_id FROM Friendship UNION SELECT user2_id,user1_id FROM Friendship);
                                    


# 1919. Leetcodify Similar Friends
select distinct l1.user_id as user1_id, l2.user_id as user2_id
from Listens l1
inner join Listens l2
on l1.day = l2.day and l1.song_id = l2.song_id and l1.user_id <> l2.user_id
where (l1.user_id,  l2.user_id) in (	select user1_id, user2_id
										from Friendship 
									)
group by l1.user_id, l1.day
having count(distinct l1.song_id) >= 3;

# Make sure have distinct -- There is no primary key for this table. It may contain duplicates.
# other options
SELECT DISTINCT user1_id, user2_id
FROM (
	SELECT user1_id, user2_id, l1.song_id, l1.day
	FROM Friendship f LEFT JOIN Listens l1 on f.user1_id = l1.user_id 
					LEFT JOIN Listens l2 on f.user2_id = l2.user_id  
	WHERE l1.song_id = l2.song_id and l1.day = l2.day
) t 
GROUP BY user1_id, user2_id, day
HAVING COUNT(DISTINCT song_id)>=3;



# 1934. Confirmation Rate
select s.user_id, 
ifnull(round(count(if(c.action='confirmed', 1, 0))/count(c.time_stamp)), 0.00) as confirmation_rate
from Signups s
left join Confirmations c using(user_id)
group by s.user_id;

# other options
# round( , 2)
select Signups.user_id, ifnull(round(sum(action = 'confirmed') / count(*), 2), 0.00) as confirmation_rate
    from Signups
    left join Confirmations
    on Signups.user_id = Confirmations.user_id
    group by Signups.user_id;
    
# sum(action = '')/ count(*)
# count(*) or count(1)
SELECT user_id
	, ROUND(IFNULL(sum(action='confirmed'),0)/count(1),2) confirmation_rate
FROM Signups s LEFT JOIN Confirmations c USING(user_id)
GROUP BY user_id;



# 1939. Users That Actively Request Confirmation Messages
# within a 24-hour window. Two messages exactly 24 hours apart are considered to be within the window. 
# check two datetime: TIMESTAMPDIFF(unit,ts1,ts2) BETWEEN 1 AND 86400;
SELECT  DISTINCT c1.user_id
FROM Confirmations c1 INNER JOIN Confirmations c2 on 
	c1.user_id=c2.user_id and TIMESTAMPDIFF(SECOND,c1.time_stamp,c2.time_stamp) BETWEEN 1 AND 86400;



# 1949. Strong Friendship
# other option  make sense
with temp as (
select user1_id, user2_id 
from Friendship
union 
select user2_id user1_id, user1_id user2_id
from Friendship
)

select f1.user1_id, f1.user2_id, count(c.user2_id) as common_friend
from Friendship f1 
join temp b 
on f1.user1_id = b.user1_id # u1 friends
join temp c 
on f1.user2_id = c.user1_id # u2 friends
and b.user2_id = c.user2_id # u1 u2 common friends
group by f1.user1_id, f1.user2_id
having count(c.user2_id) >= 3 ;

# other option better but slightly harder to do
with friends as (
    select user1_id,
           user2_id
    from friendship
    union
    select user2_id,
           user1_id
    from friendship
)

SELECT 
    A.USER1_ID AS user1_id,
    B.USER1_ID AS user2_id,
    COUNT(*) AS common_friend 
FROM 
    friends A
INNER JOIN 
    friends B
ON A.USER2_ID=B.USER2_ID AND A.USER1_ID<B.USER1_ID
WHERE (A.user1_id,B.user1_id) IN (SELECT user1_id,user2_id FROM friendship)
GROUP BY 1,2
HAVING COUNT(*)>2;



# 1951. All the Pairs With the Maximum 
select user1_id, 
	user2_id
from (
select r1.user_id as user1_id, 
	r2.user_id as user2_id,
    count(r1.follower_id) as cnt
from Relations r1
join Relations r2
using(follower_id)
where r1.user_id < r2.user_id
group by r1.user_id, r2.user_id
) as temp
where cnt = (select max(cnt) from r);

# modify
select r1.user_id as user1_id, 
	r2.user_id as user2_id,
    count(r1.follower_id) as cnt
from Relations r1
join Relations r2
using(follower_id)
where r1.user_id < r2.user_id
group by r1.user_id, r2.user_id
having count(r1.follower_id) = max(count(r1.follower_id));



# 1965. Employees With Missing Information
# the trick part is union and not in
select employee_id
from Employees
where employee_id not in (select distinct employee_id from Salaries)
union
select employee_id
from Salaries
where employee_id not in (select distinct employee_id from Employees)
order by employee_id asc;

select s.employee_id
from Employees e
right join Salaries s
on s.employee_id = e.employee_id
where e.name is null
union
select e.employee_id
from Employees e
left join Salaries s
on s.employee_id = e.employee_id
where s.salary is null 
order by employee_id;



# 1972. First and Last Call On the Same Day
with t1 as (
	select caller_id as user_id, call_time, rank() over(partition by date(call_time) order by call_time asc) as rnk
	from Calls
	union
	select recipient_id as user_id, call_time, rank() over(partition by date(call_time) order by call_time asc) as rnk
	from Calls
    order by call_time
) ,
t2 as(
	select *, rank() over(partition by date(call_time) order by call_time desc) as rnk2
    from t1
)

select distinct user_id 
from t2
where rnk = 1 and rnk2 = 1

# other options
with t1 as (
    select caller_id as id1, recipient_id as id2, call_time from Calls
    union all
    select recipient_id as id1, caller_id as id2, call_time from Calls
),
t2 as (
    select id1, id2, date(call_time) as dt,
        rank() over(partition by id1, date(call_time) order by call_time) as rk1,
        rank() over(partition by id1, date(call_time) order by call_time desc) as rk2
    from t1
)
select distinct id1 user_id from t2 
where rk1 = 1 or rk2 = 1
group by dt, id1 having count(distinct id2) = 1;

# other options 2
WITH CTE AS (
        SELECT caller_id AS user_id, call_time, recipient_id FROM Calls
        UNION 
        SELECT recipient_id AS user_id, call_time, caller_id AS recipient_id FROM Calls
            ),

CTE1 AS (
        SELECT 
        user_id,
        recipient_id,
        DATE(call_time) AS DAY,
        DENSE_RANK() OVER(PARTITION BY user_id, DATE(call_time) ORDER BY call_time ASC) AS RN,
        DENSE_RANK() OVER(PARTITION BY user_id, DATE(call_time) ORDER BY call_time DESC) AS RK
        FROM CTE
        )

SELECT DISTINCT user_id
FROM CTE1
WHERE RN = 1 OR RK = 1
GROUP BY user_id, DAY
HAVING COUNT(DISTINCT recipient_id) = 1



# 1978. Employees Whose Manager Left the Company
select employee_id
from Employees
where salary < 30000 and manager_id not in (select employee_id from Employees)
order by employee_id;



# 1988. Find Cutoff Score for Each School
select school_id, if(student_count < capacity, min(score), -1) as score
from Schools, Exam;

# other options
select school_id, ifnull(min(score), -1) as score
from Schools left join Exam
on capacity >= student_count
group by school_id;



# 1990. Count the Number of Experiments
# this would not work
select platform,
experiment_name, 
ifnull(count(experiment_id) over(partition by platform, experiment_name),0) as num_experiments
from Experiments;

# other options have to use cross join
select p.platform, 
n.experiment_name, 
ifnull(count(e.platform), 0) as num_experiments 
from (
    select 'Android' as platform
    union all
    select 'IOS' as platform
    union all
    select 'Web' as platform
) as p
cross join (
    select 'Reading' as experiment_name
    union all
    select 'Sports' as experiment_name
    union all
    select 'Programming' as experiment_name
) as n
left join Experiments e
using (platform, experiment_name)
group by p.platform, n.experiment_name;



# 2004. The Number of Seniors and Juniors to Join the Company
with t1 as(
select *,
	sum(salary) over(partition by experience, order by salary) as total_salary
    from Candidates
),
t2 as(
	select * 
    from t1
	where total_salary <= 70000 
    and experience = 'Senior'
),
t3 as(
	select * 
    from t1
	where total_salary <= (70000 - (select ifnull(max(total_salary), 0) from t2))  
    and experience = 'Junior'
)

select 'Junior' as Experience,
	count(employee_id) as accepted_candidates
from t1
union
select 'Senior' as Experience,
	count(employee_id) as accepted_candidates
from t2

# other options
WITH CTE AS (
    SELECT 
        employee_id, 
        experience, 
        SUM(salary) OVER(PARTITION BY experience ORDER BY salary,employee_id ASC) AS RN 
    FROM Candidates
)
      
SELECT 
    'Senior' AS experience, 
    COUNT(employee_id) AS accepted_candidates 
FROM CTE 
WHERE experience = 'Senior' AND RN < 70000
UNION
SELECT 
    'Junior' AS experience, 
    COUNT(employee_id) AS accepted_candidates 
FROM CTE 
WHERE experience = 'Junior' AND RN < (SELECT 70000 - IFNULL(MAX(RN),0) FROM CTE WHERE experience = 'Senior' AND RN < 70000)
 


# 2010. The Number of Seniors and Juniors to Join the Company II
with tmp as (
	select *,
    sum(salary) over(partition by experience order by salary) as total_salary
    from Candidates
)
select employee_id from tmp where experience = 'Senior' and total_salary <= 70000 
union
select employee_id from tmp where experience = 'Junior' 
and total_salary <= (70000 - (select ifnull(max(total_salay), 0) 
								from tmp 
                                where experience = 'Senior' and total_salary <= 70000))
                                
# other options 
WITH GET_SENIOR AS(
SELECT * FROM (SELECT EMPLOYEE_ID, 70000-SUM(SALARY) OVER(ORDER BY SALARY) AS BUDGET_LEFT FROM CANDIDATES
WHERE EXPERIENCE='Senior') AS A
WHERE BUDGET_LEFT>0),


GET_JUNIOR AS(
    SELECT * FROM (SELECT employee_id, (SELECT COALESCE(MIN(BUDGET_LEFT),70000) FROM GET_SENIOR)-SUM(SALARY) OVER(ORDER BY SALARY) AS BUDGET_LEFT FROM CANDIDATES
WHERE EXPERIENCE='Junior') AS B
WHERE BUDGET_LEFT>0
)

SELECT * FROM (
SELECT employee_id FROM GET_JUNIOR
UNION 
SELECT employee_id FROM GET_SENIOR) AS A



# 2020. Number of Accounts That Did Not Stream
select count(distinct account_id) as accounts_count
from Subscriptions s1
left join Streams s2 using(account_id)
where year(end_date) = 2021 and year(stream_date) <> 2021;

# other options
SELECT COUNT(account_id) AS accounts_count  # find counts
FROM Subscriptions
WHERE YEAR(start_date) <= 2021 AND YEAR(end_date) >= 2021 # has subscription in 2021
	AND account_id NOT IN (SELECT account_id FROM Streams WHERE YEAR(stream_date) = '2021') ; # has no streams in 2021



# 2026. Low-Quality Problems



# 2041. Accepted Candidates From the Interviews
select candidate_id
from Candidate_id
where years_of_exp >= 2 and interview_id in (select interview_id 
											from Rounds 
                                            group by interview_id
                                            having sum(score) > 15);
                                            
select candidate_id
from Candidate_id
left join Rounds
using(interview_id)
where years_of_exp >= 2 
group by candidate_id, interview_id
having sum(score) > 15;



# 2051. The Category of Each Member in the Store
# between and
select member_id,
	name,
	case when (100 * ifnull(count(p.visit_id),0) / ifnull(count(v.member_id),0)) >= 80 then 'Diamond' 
    case when (100 * ifnull(count(p.visit_id),0) / ifnull(count(v.member_id),0)) between 50 and 79 then 'Gold' 
    case when (100 * ifnull(count(p.visit_id),0) / ifnull(count(v.member_id),0)) between 1 and 49 then 'Silver'
    case when (100 * ifnull(count(p.visit_id),0) / ifnull(count(v.member_id),0)) == 0 then 'Bronze' end as category
from Members m
left join Visits v on m.member_id = v.member_id
left join Purchases p on p.visit_id = v.visit_id
group by member_id, name

# other option
WITH CTE AS (
    SELECT MEMBER_ID, COUNT(B.VISIT_ID)*100/COUNT(A.VISIT_ID) AS SCORE FROM VISITS A
    LEFT JOIN PURCHASES B 
    ON A.VISIT_ID=B.VISIT_ID
    GROUP BY MEMBER_ID
)

SELECT A.*, 
case when SCORE>=80 then "Diamond"
when SCORE>=50 and SCORE<80 then "Gold"
when SCORE <50 then "Silver"
WHEN SCORE IS NULL THEN "Bronze" end as category
FROM MEMBERS A
LEFT JOIN CTE B
ON A.MEMBER_ID=B.MEMBER_ID
ORDER BY 1



# 2066. Account Balance
select account_id, 
	day, 
    sum(if(type='Deposit', +amount, -amount)) over(partition by account_id order by day) as balance
from Transactions
order by account_id, day



# 2072. The Winner University



# 2082. The Number of Rich Customers



# 2084. Drop Type 1 Orders for Customers With Type 0 Orders



# 2112. The Airport With the Most Traffic



# 2118. Build the Equation



# 2142. The Number of Passengers in Each Bus I
# lag(return_value , offset, default)
# BETWEEN AND clause is lower and upper limit inclusive. can't use at here
select bus_id,
 sum(if(p.arrival_time <= t.arrival_time and p.arrival_time > pre_arrival) , 1, 0)) over(partition by t.arrival_time order by p.arrival_time) ) as passengers_cnt
from (
	select *,
    lag(arrival_time, 1, 0) over(order by bus_id) as pre_arrival
	from Buses
	) as temp t
join Passengers p
on t.arrival_time >= p.arrival_time and p.arrival_time > pre_arrival



# 2153. The Number of Passengers in Each Bus II



# 2159. Order Two Columns Independently



# 2173. Longest Winning Streak



# 2175. The Change in Global Rankings
select team_id,
name,
rka - rk as rank_diff
from ( 
	select team_id,
	name, 
	row_number() over(order by points desc, name) as rk
    row_number() over(order by (points + points_change) desc, name) as rka
	from TeamPoints t
	join PointsChange pc using(team_id)
) as temp

# other options
# Calculate ranks before and after the changes
# cast( as signed)
with ranks as 
(select 
    T.team_id, 
    T.name, 
    rank() over (order by points desc, name) as rank_old,
    rank() over (order by points + points_change desc, name) as rank_new
from TeamPoints T 
join PointsChange P using (team_id))

## Report rank_diff
select 
    team_id,
    name,
    (cast(rank_old as signed) - cast(rank_new as signed)) as rank_diff
from ranks


# 2199. Finding the Topic of Each Post
# REGEXP_LIKE(expr, pat[, match_type])
# select Value from DemoTable where 'Relational' LIKE concat('%',Value,'%');
# GROUP_CONCAT()
SELECT
  P.post_id,
  COALESCE(GROUP_CONCAT(DISTINCT K.topic_id ORDER BY K.topic_id), 'Ambiguous!') as topic
FROM
  Posts P
  LEFT JOIN Keywords K 
  ON REGEXP_LIKE(P.content, CONCAT('\\b', K.word, '\\b'), 'i')
GROUP BY
  P.post_id;



# 2205. The Number of Users That Are Eligible for Discount



# 2228. Users With Two Purchases Within Seven Days
# date_add(, interval 7 day)
select distinct user_id
from(
select *, lead(purchase_date,1) over(partition by user_id order by purchase_date) as next_per
from Purchases
) as temp
where date_add(purchase_date, interval 7 day) <= next_per
order by user_id;

-- METHOD 1
# RANGE BETWEEN CURRENT ROW AND INTERVAL 7 DAY FOLLOWING
SELECT DISTINCT USER_ID FROM 
	(SELECT 
    USER_ID,
    COUNT(PURCHASE_ID) OVER(PARTITION BY USER_ID ORDER BY PURCHASE_DATE RANGE BETWEEN CURRENT ROW AND INTERVAL 7 DAY FOLLOWING) AS CNT 
FROM PURCHASES) AS A
WHERE CNT=2

-- METHOD 2
# DATEDIFF(NEXT_SHOPPIN,PURCHASE_DATE)<=7
WITH CTE AS(
    SELECT DISTINCT USER_ID,PURCHASE_DATE, 
LEAD(PURCHASE_DATE,1) OVER(PARTITION BY USER_ID ORDER BY PURCHASE_DATE) AS NEXT_SHOPPIN,
DATE_ADD(PURCHASE_DATE, INTERVAL 7 DAY) AS NEXT_7_DAY FROM PURCHASES)

SELECT USER_ID FROM CTE WHERE
DATEDIFF(NEXT_SHOPPIN,PURCHASE_DATE)<=7



# 2230. The Users That Are Eligible for Discount



# 2238. Number of Times a Driver Was a Passenger



# 2252. Dynamic Pivoting of a Table



# 2253. Dynamic Unpivoting of a Table



# 2292. Products With Three or More Or


