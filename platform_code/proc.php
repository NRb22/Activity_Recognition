<?php
if(!$_GET['no'])
{
	$_GET['no'] = 0;
}
$data = array();
$no = $_GET['no'];
$status = false;

$db = new mysqli('localhost', 'root', '', 'learningEyes');
//$db->query('SET NAMES utf8');

for($i=0; $i<80; $i++)
{
	$status = false;
	$res = $db->query('SELECT * FROM detect_log WHERE no > "' . $_GET['no'] . '"');

	if($res->num_rows > 0)
	{		
		while($v = $res->fetch_array(MYSQLI_ASSOC))
		{
			$data[] = $v;
			$no = $v['no'];
			$status = true;

		}
		
		break;
	}

	usleep(250000);
}

echo json_encode(array('no' => $no, 'data' => $data, 'status' => $status));
?>
